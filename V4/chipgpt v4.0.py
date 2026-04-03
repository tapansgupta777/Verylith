import os
import json
import argparse
import subprocess
import hashlib
import re
import time
import random
import shutil
import concurrent.futures
from collections import OrderedDict
from openai import OpenAI
from groq import Groq

# ==============================================================================
# MODEL TIER CONFIGURATION
# ==============================================================================
FAST_MODEL   = "gpt-4o-mini"
STRONG_MODEL = "gpt-5-mini"

REASONING_MODELS = {
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "gpt-5-mini",
}

def _is_reasoning_model(model_name: str) -> bool:
    return model_name.strip() in REASONING_MODELS

MAX_TOKENS_FAST   = 4096
MAX_TOKENS_STRONG = 16384   # kept at 16384 — gpt-5-mini needs the space

# ==============================================================================
# LRU CACHE
# ==============================================================================
import threading

class BoundedCache:
    """True LRU cache — thread-safe, prevents memory explosion on long runs."""
    def __init__(self, max_size=100):
        self.cache    = OrderedDict()
        self.max_size = max_size
        self._lock    = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        return None

    def put(self, key, val):
        with self._lock:
            self.cache[key] = val
            self.cache.move_to_end(key)
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def __contains__(self, key):
        with self._lock:
            return key in self.cache

ARCH_CACHE = BoundedCache(50)
RTL_CACHE  = BoundedCache(200)
TB_CACHE   = BoundedCache(200)

# ==============================================================================
# OPENAI CLIENT
# ==============================================================================

def get_openai_client(timeout_s: int = 180) -> OpenAI:
    """Create an OpenAI client with the specified timeout.
    Stateless — each call creates a fresh client so the timeout is always respected.
    OpenAI clients are lightweight (just httpx session config), so this is fine.
    """
    return OpenAI(timeout=timeout_s)

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def code_hash(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)
    normalized = re.sub(r'\s+', ' ', text.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def get_cache_key(*args) -> str:
    combined = "".join(code_hash(str(a)) for a in args)
    return code_hash(combined)

def truncate_log(log: str, max_lines: int = 15) -> str:
    lines = log.strip().split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + "\n... [ERRORS TRUNCATED]"
    return log

def extract_primary_error(log: str) -> str:
    lines = log.strip().split('\n')
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in
               ("error", "mismatch", "fail", "syntax", "%warning", "assertion failed")):
            start = max(0, i - 1)
            end   = min(len(lines), i + 3)
            return "\n".join(lines[start:end])
    return truncate_log(log, 5)

def classify_failure(log: str) -> str:
    l = log.lower()
    if "timeout" in l or "simulation timeout" in l:
        return "TB_TIMEOUT"
    if "unoptflat" in l or ("loop" in l and "comb" in l):
        return "COMBINATIONAL_LOOP"
    if "compile" in l or "syntax" in l:
        return "SYNTAX_ERROR"
    if "latch" in l:
        return "LATCH_INFERRED"
    if "width" in l or "size" in l:
        return "WIDTH_MISMATCH"
    # X/Z propagation: match standalone x/z values, not hex prefixes like 0x01
    if re.search(r"(?<![0-9a-fA-F'hH])x(?![0-9a-fA-F])|(?<![0-9a-fA-F'hH])z(?![0-9a-fA-F])", l) or re.search(r'\bxx+\b', l):
        return "X_PROPAGATION_BUG"
    if "stuck" in l or "never changes" in l:
        return "STATE_MACHINE_STUCK"

    m = re.search(
        r'expected\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+).*?got\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+)', l
    )
    if m:
        try:
            g1, g2 = m.group(1), m.group(2)
            # Skip if values contain x/z (unknown/high-Z), not valid numbers
            if re.search(r'[xzXZ]', g1 + g2):
                pass
            else:
                has_hex = ('0x' in l or any(c in 'abcdef'
                           for c in g1.lower() + g2.lower()))
                base = 16 if has_hex else 10
                exp  = int(g1, base)
                got  = int(g2, base)
                if abs(exp - got) == 1:
                    return "OFF_BY_ONE"
        except (ValueError, OverflowError):
            pass

    if "priority" in l or "instead of" in l or "order" in l:
        return "PRIORITY_ORDER_BUG"
    if "shift" in l or "serial" in l or "sipo" in l or "siso" in l or "piso" in l:
        return "SHIFT_LOGIC"
    if "increment" in l or "count up" in l or "count_up" in l:
        return "INCREMENT_LOGIC"
    if "decrement" in l or "count down" in l or "count_down" in l:
        return "DECREMENT_LOGIC"
    if "overflow" in l:
        return "OVERFLOW_FLAG"
    if "underflow" in l:
        return "UNDERFLOW_FLAG"
    if "reset" in l:
        return "RESET_LOGIC"
    if "load" in l:
        return "LOAD_LOGIC"
    if "assert" in l or "assertion" in l or "invariant" in l:
        return "PROTOCOL_ASSERTION_FAILED"
    if re.search(r'(expected|mismatch|fail)', l):
        return "LOGIC_MISMATCH"
    return "TB_STRUCTURAL"

def get_error_fingerprint(failure_type: str, full_log: str) -> str:
    primary   = extract_primary_error(full_log)
    line_nums = re.findall(r'\.v:(\d+):', primary)
    line_sig  = "_".join(line_nums[:3]) if line_nums else code_hash(primary)[:8]
    return f"{failure_type}_{line_sig}"

def localize_bug(v_code: str, failure_type: str, error_log: str = "") -> str:
    """Extract RTL lines most likely to contain the bug.

    Strategy 1: Parse the failing signal name from the error log and find its
    driver in the RTL — this gives the exact assignment that needs to change.
    Strategy 2: Fall back to keyword search by failure type.
    Returns a compact snippet for the mutation prompt.
    """
    lines    = v_code.split('\n')
    snippets = []

    # Strategy 1: extract signal name from error log
    # Try multiple patterns from most-specific to least-specific:
    #   "expected s_ready=0"  →  group 1 = s_ready
    #   "expected parallel_out"  →  group 1 = parallel_out
    #   "FAIL: count mismatch"  →  group 1 = count
    if error_log:
        signal = None
        for pattern in [
            r'(?:expected|got)\s+(\w+)\s*[=!<>]',          # expected signal=val
            r'(?:expected|got)\s+(\w+)\s+(?:0x|8\'|[0-9])',# expected signal 0x..
            r'\bFAIL[^:]*:\s*\w+\s+(\w+)\s+',              # FAIL: msg signal ...
            r'(?:expected|got)\s+(\w+)',                     # expected signal (bare)
        ]:
            m = re.search(pattern, error_log, re.IGNORECASE)
            if m:
                candidate = m.group(1).lower()
                # Skip generic non-signal words
                if candidate not in ('0', '1', 'true', 'false', 'high', 'low',
                                     'test', 'after', 'at', 'when', 'got', 'expected',
                                     'fail', 'pass', 'error', 'mismatch', 'assertion'):
                    signal = candidate
                    break

        if signal:
            for i, line in enumerate(lines):
                if re.search(
                    rf'\bassign\s+{re.escape(signal)}\b'
                    rf'|\b{re.escape(signal)}\s*<='
                    rf'|\b{re.escape(signal)}\s*=(?!=)',
                    line, re.IGNORECASE
                ):
                    start = max(0, i - 1)
                    end   = min(len(lines), i + 3)
                    snippets.append("\n".join(lines[start:end]).strip())

    # Strategy 2: keyword fallback by failure type
    if not snippets:
        keywords = []
        if failure_type == "INCREMENT_LOGIC":       keywords = ["+ 1", "+1", "increment", "add"]
        elif failure_type == "DECREMENT_LOGIC":     keywords = ["- 1", "-1", "decrement", "sub"]
        elif failure_type == "OVERFLOW_FLAG":       keywords = ["overflow", "255", "hff", "== 255"]
        elif failure_type == "UNDERFLOW_FLAG":      keywords = ["underflow", "== 0"]
        elif failure_type == "RESET_LOGIC":         keywords = ["rst", "reset"]
        elif failure_type == "LOAD_LOGIC":          keywords = ["load"]
        elif failure_type == "PRIORITY_ORDER_BUG":  keywords = ["if", "else if"]
        elif failure_type == "SHIFT_LOGIC":         keywords = ["shift", "serial", "{"]
        elif failure_type == "OFF_BY_ONE":          keywords = ["<", "<=", ">=", "=="]
        elif failure_type in ("LOGIC_MISMATCH", "PROTOCOL_ASSERTION_FAILED"):
            keywords = ["assign", "ready", "valid", "buffer"]
        elif failure_type in ("STATE_MACHINE_STUCK",):
            keywords = ["always", "state", "next_state"]

        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in keywords):
                start = max(0, i - 1)
                end   = min(len(lines), i + 2)
                snippets.append("\n".join(lines[start:end]).strip())

    if not snippets:
        return ""

    unique_snippets = list(dict.fromkeys(snippets))
    return "...\n" + "\n...\n".join(unique_snippets[:4]) + "\n..."


def build_delta_mutation(v_code: str, failure_type: str, error_log: str) -> str:
    """Build a focused delta snippet for the mutation prompt.

    Instead of sending the full RTL, extract just the failing driver lines
    and the exact assertion that failed. This tightens the LLM's attention
    to the specific change needed.
    """
    if not error_log:
        return ""

    # Extract the first FAIL line as the specific assertion
    fail_line = ""
    for line in error_log.splitlines():
        if re.search(r'\bFAIL\b|\bfail\b|expected.*got|mismatch', line, re.I):
            fail_line = line.strip()
            break

    localized = localize_bug(v_code, failure_type, error_log)
    if not fail_line and not localized:
        return ""

    parts = []
    if fail_line:
        parts.append(f"FAILING ASSERTION:\n  {fail_line}")
    if localized:
        parts.append(f"DRIVER CODE TO FIX:\n{localized}")

    return "\n\n".join(parts)

def parse_llm_json(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        print("   ⚠️  LLM returned empty response.")
        return {}
    try:
        triple_tick = "`" * 3
        clean = re.sub(
            rf'{triple_tick}(?:json)?\s*|{triple_tick}\s*$', '',
            raw_text.strip(), flags=re.MULTILINE
        ).strip()
        clean = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', clean)
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group(), strict=False)
        return json.loads(clean, strict=False)
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON Parse Error: {e}")
        return {}

def clean_code_string(raw_code: str) -> str:
    if not raw_code:
        return ""
    clean = raw_code.strip()
    triple_tick = "`" * 3
    for fence in (f"{triple_tick}verilog", f"{triple_tick}systemverilog",
                  f"{triple_tick}v", triple_tick):
        clean = clean.replace(fence, "")
    return clean.strip()

# ==============================================================================
# CORE LLM ROUTER
# ==============================================================================
def run_llm(
    prompt:           str,
    system_prompt:    str,
    provider:         str,
    max_api_retries:  int  = 3,
    cand_idx:         int  = 0,
    force_fast_model: bool = False,
    silent:           bool = False,
) -> dict:
    temps   = [0.2, 0.5, 0.8, 1.0]
    efforts = ["low", "medium", "high"]

    model_name   = FAST_MODEL if force_fast_model else STRONG_MODEL
    is_reasoning = _is_reasoning_model(model_name)
    max_tokens   = MAX_TOKENS_FAST if force_fast_model else MAX_TOKENS_STRONG

    for attempt in range(max_api_retries):
        try:
            if provider == "openai":
                client    = get_openai_client(timeout_s=180 if not force_fast_model else 90)
                temp_idx  = min(cand_idx + attempt, len(temps) - 1)
                kwargs    = {
                    "model":    model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                }
                if is_reasoning:
                    kwargs["max_completion_tokens"] = max_tokens
                    kwargs["reasoning_effort"]      = efforts[temp_idx % 3]
                    # Reasoning models don't support response_format=json_object.
                    # Inject a mandatory JSON instruction into the system prompt instead.
                    for msg in kwargs["messages"]:
                        if msg["role"] == "system":
                            msg["content"] = (
                                msg["content"].rstrip()
                                + "\n\nCRITICAL: Your response MUST be valid JSON only. "
                                "Output nothing before or after the JSON object. "
                                "No markdown fences, no explanation, no preamble."
                            )
                            break
                else:
                    kwargs["max_tokens"]      = max_tokens
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["temperature"]     = temps[temp_idx]

                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling {model_name} (Attempt {attempt+1})...")

                response = client.chat.completions.create(**kwargs)
                content  = response.choices[0].message.content

                if not content or not content.strip():
                    print(f"      ⚠️  Thread {cand_idx+1}: Empty content. Retrying...")
                    continue

                return parse_llm_json(content)

            elif provider == "groq":
                client   = Groq()
                temp_idx = min(cand_idx + attempt, len(temps) - 1)
                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling Groq llama-3.3-70b...")
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=temps[temp_idx],
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    timeout=90,
                )
                return parse_llm_json(response.choices[0].message.content)

        except Exception as e:
            err = str(e).lower()
            if "400" in err and ("unsupported parameter" in err or "invalid_request" in err):
                print(f"   ❌ Thread {cand_idx+1}: Bad request: {e}")
                return {}
            elif "429" in err or "quota" in err:
                wait = 35 * (attempt + 1)
                print(f"   ⏳ Thread {cand_idx+1}: Rate limit. Sleeping {wait}s...")
                time.sleep(wait)
            elif "timeout" in err:
                print(f"   ⚠️  Thread {cand_idx+1}: API timeout. Retrying...")
            else:
                print(f"   ⚠️  Thread {cand_idx+1}: Fatal API error: {e}")
                time.sleep(5)

    return {}

# ==============================================================================
# AGENT: CLARIFIER
# ==============================================================================
def generate_specification(prompt: str, provider: str) -> dict:
    print(f"   📝 Routing to CLARIFIER AGENT ({provider.upper()}) [{FAST_MODEL}]...")
    system_prompt = """
You are a Lead Hardware Specifications Engineer and an Encyclopedic Hardware Knowledge Base.
Translate the user's ambiguous request into a strict, disambiguated technical specification.

CRITICAL GENERALIZATION RULE:
If the user asks for a standard component (e.g., 'Skid Buffer', 'Async FIFO', 'Arbiter',
'UART', 'SPI', 'AXI', 'shift register', 'counter'), DO NOT just restate their prompt.
You MUST aggressively expand `formalized_request` using your internal knowledge:
  1. List the textbook logic equations and architectural structures required.
  2. Name every required internal register (e.g., buffer_data, buffer_valid, read_ptr).
  3. State rigid protocol invariants explicitly:
     - For ready/valid: "valid may not deassert until ready is high (once valid asserts,
       data must hold until accepted). s_ready must deassert when internal buffer is full
       and downstream is not consuming."
     - For FIFO: "full flag prevents writes, empty flag prevents reads."
  4. Describe the exact combinational conditions for every output flag.
This expansion is critical — downstream agents must not guess implicit dependencies.

Return pure JSON:
{
  "formalized_request": "...",
  "data_widths": "...",
  "sequential_or_combinational": "...",
  "clocking_scheme": "...",
  "reset_behavior": "...",
  "overflow_behavior": "...",
  "protocol_invariants": "...",
  "required_internal_registers": "..."
}
"""
    return run_llm(prompt, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)

# ==============================================================================
# AGENT: ARCHITECT
# ==============================================================================
ARCH_STYLES = [
    "behavioral_single_always_block",
    "explicit_datapath_and_controller",
    "pipelined_registered_stages",
    "decoupled_control_and_status",
    "register_chain_optimized",
    "onehot_fsm_encoded",
    "registered_output_buffered",
]

def generate_architecture(
    spec:              dict,
    provider:          str,
    previous_failures: str = None,
    attempt_idx:       int = 0,
    forced_style:      str = None,
) -> dict:
    cache_key = get_cache_key(json.dumps(spec), previous_failures, attempt_idx, forced_style)
    if cache_key in ARCH_CACHE:
        print(f"   ⚡ Cache Hit: Reusing Micro-Architecture blueprint.")
        return ARCH_CACHE.get(cache_key)

    print(f"   📐 Routing to MICRO-ARCHITECT AGENT ({provider.upper()}) [{FAST_MODEL}]...")
    error_context = ""
    if previous_failures:
        error_context = f"""
CRITICAL FEEDBACK FROM PREVIOUS ARCHITECTURES:
{previous_failures}
Do NOT repeat the exact same datapath, pipelining, or mux structures.
"""

    style_directive = (
        f"\nYou MUST strictly adhere to the `{forced_style}` implementation paradigm.\n"
        if forced_style else ""
    )

    salt = random.randint(10000, 99999)
    system_prompt = f"""
You are an elite SoC Micro-Architect. Design a strictly synthesizable hardware architecture.
{error_context}{style_directive}

CRITICAL CLASSIFICATION:
- DATAPATH: counters, FIFOs, ALUs, datapaths, memories, shift registers, arithmetic units.
- FSM: protocol controllers, sequence detectors, handshake controllers, state machines.
A counter with overflow/underflow flags is always DATAPATH, never FSM.

You MUST list ALL ports from the specification in port_interface.

ALGORITHM DESCRIPTION REQUIREMENT:
Describe the behavioral algorithm at pseudocode level. For every register, give its exact
next-state logic. For every output flag, give its exact combinational condition.
Use `control_logic`, `flag_logic`, and `priority_order` fields.

Return pure JSON exactly matching this schema:
{{
    "module_class": "DATAPATH",
    "template_type": "counter",
    "implementation_style": "{forced_style or 'behavioral_single_always_block'}",
    "clock_and_reset": {{"clock": "clk", "reset": "rst_n", "active_low": true}},
    "port_interface": [{{"name": "clk", "direction": "input", "width": 1}}],
    "registers": [{{"name": "count", "width": 8, "reset_value": "0"}}],
    "memory_blocks": [],
    "datapath_nodes": [
        {{"op": "+", "dst": "next_count", "a": "count", "b": "1"}}
    ],
    "muxes": [
        {{
            "dst": "count_next_final",
            "control_conditions": [
                {{"when": "load_en", "route": "load_data"}},
                {{"default": "count"}}
            ]
        }}
    ],
    "control_logic": [
        {{
            "register": "count",
            "algorithm": "if load_en → count = load_data; else if enable and up_down → count = count + 1; else if enable and !up_down → count = count - 1; else count unchanged"
        }}
    ],
    "flag_logic": [
        {{"signal": "overflow", "type": "combinational", "condition": "count == 255 AND enable AND up_down"}}
    ],
    "priority_order": ["rst_n (async, active-low)", "load_en", "enable"],
    "fsm_specific": {{"state_encoding": {{}}, "next_state_logic": []}}
}}

[VARIATION SALT: {salt}]
"""
    result = run_llm(json.dumps(spec), system_prompt, provider,
                     cand_idx=attempt_idx, force_fast_model=True, silent=True)
    if result:
        ARCH_CACHE.put(cache_key, result)
    return result

def validate_architecture(arch: dict, expected_min_ports: int = 3) -> bool:
    if not arch or "module_class" not in arch:
        return False
    if not arch.get("clock_and_reset"):
        return False
    ports = arch.get("port_interface", [])
    if not ports or len(ports) < expected_min_ports:
        return False
    port_names = [p.get("name", "").lower() for p in ports]
    if not any("clk" in n or "clock" in n for n in port_names):
        return False
    if not any("rst" in n or "reset" in n for n in port_names):
        return False
    if arch["module_class"] == "FSM":
        fsm = arch.get("fsm_specific", {})
        if not fsm.get("state_encoding") or not fsm.get("next_state_logic"):
            return False
    return True

# ==============================================================================
# AGENT: REVIEWER
# ==============================================================================
def review_hardware(v_code: str, provider: str) -> dict:
    system_prompt = """
You are an elite Silicon Verification Engineer.
Analyze this Verilog RTL strictly for:
  1. Inferred latches (missing defaults in combinational always blocks)
  2. Missing reset conditions on sequential registers
  3. Combinational feedback loops

CRITICAL RULES:
  - DO NOT rewrite the module. Output only a minimal patch.
  - NEVER convert assign statements to reg assignments.
  - If the code is clean, return {"status": "PASSED", "fixed_code": ""}.
  - If patching, return {"status": "REJECTED", "fixed_code": "<full corrected module>"}.
Return pure JSON: {"status": "...", "fixed_code": "..."}
"""
    return run_llm(v_code, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)

# ==============================================================================
# RTL SKELETON LIBRARY
# These are verified-correct structural skeletons for common design types.
# They are NOT used as format()-string templates (Verilog braces would conflict).
# Instead they are injected into the RTL generator prompt as a "starting skeleton"
# that the LLM must preserve structurally and only fill with the correct parameters.
# ==============================================================================

# Keys must match detect_design_type() return values
RTL_SKELETONS = {
    "counter": """\
// SKELETON — preserve structure, fill WIDTH and signal names from spec
module <name> #(parameter WIDTH = 8) (
    input clk, input rst_n,
    input enable,
    output reg [WIDTH-1:0] count
);
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) count <= 0;
    else if (enable) count <= count + 1;
end
endmodule""",

    "shift_register": """\
// SKELETON — preserve structure, fill WIDTH and direction from spec
module <name> #(parameter WIDTH = 8) (
    input clk, input rst_n,
    input shift_en, input serial_in,
    output reg [WIDTH-1:0] parallel_out
);
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) parallel_out <= 0;
    else if (shift_en) parallel_out <= {parallel_out[WIDTH-2:0], serial_in}; // LEFT shift, LSB-in
end
endmodule""",

    "skid_buffer": """\
// SKELETON — canonical skid buffer: output register + one skid slot
// Key invariant: s_ready = !buf_valid || m_ready
// Source can always push UNLESS skid is occupied AND downstream isn't draining
module <n> #(parameter WIDTH = 8) (
    input                  clk,
    input                  rst_n,
    input                  s_valid,
    input      [WIDTH-1:0] s_data,
    output                 s_ready,
    output reg             m_valid,
    output reg [WIDTH-1:0] m_data,
    input                  m_ready
);
    reg             buf_valid;
    reg [WIDTH-1:0] buf_data;

    // s_ready: source can push when skid is empty, OR master is draining this cycle
    assign s_ready = !buf_valid || m_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_valid <= 1'b0;
            m_valid   <= 1'b0;
            m_data    <= {WIDTH{1'b0}};
            buf_data  <= {WIDTH{1'b0}};
        end else begin
            // DRAIN path: master consumed output, or output is empty
            // Refill from skid slot first; if skid empty, take directly from source
            if (m_ready || !m_valid) begin
                if (buf_valid) begin
                    m_valid   <= 1'b1;
                    m_data    <= buf_data;
                    buf_valid <= 1'b0;
                end else if (s_valid) begin
                    m_valid   <= 1'b1;
                    m_data    <= s_data;
                end else begin
                    m_valid   <= 1'b0;
                end
            end
            // PUSH path: source is pushing AND output register is busy (not draining)
            // s_ready guarantees buf_valid is 0 in this case
            if (s_valid && s_ready && !(m_ready || !m_valid)) begin
                buf_valid <= 1'b1;
                buf_data  <= s_data;
            end
        end
    end
endmodule""",

    "fsm": """\
// SKELETON — two-always FSM, fill states and transitions from spec
// State register
reg [1:0] state, next_state;
// State encoding: define states as localparams
// Combinational next-state logic
always @(*) begin
    next_state = state;
    case (state)
        // FILL: add state transitions here
    endcase
end
// Sequential state register with async reset
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) state <= 0;
    else state <= next_state;
end""",
}


def detect_design_type(spec: dict, prompt: str) -> str:
    """Classify the design type from spec fields and prompt text.
    Uses spec first (more reliable) then falls back to prompt keywords.
    Returns a key into RTL_SKELETONS, or 'generic' if no match.
    """
    # Use the formalized_request from clarifier spec if available
    text = (
        spec.get("formalized_request", "") + " " + prompt
    ).lower()

    # Order matters — more specific patterns first
    if any(kw in text for kw in ("skid buffer", "skid_buffer", "ready/valid pipeline",
                                  "pipeline register", "ready valid pipeline")):
        return "skid_buffer"
    if any(kw in text for kw in ("shift register", "sipo", "piso", "siso", "serial-in",
                                  "serial in parallel", "shift reg")):
        return "shift_register"
    if any(kw in text for kw in ("up/down counter", "up down counter", "bidirectional counter",
                                  "loadable counter", "saturating counter")):
        return "counter"
    if "counter" in text and "fifo" not in text:
        return "counter"
    if any(kw in text for kw in ("state machine", "fsm", "sequence detector",
                                  "protocol controller")):
        return "fsm"
    return "generic"


def get_rtl_skeleton(design_type: str, design_name: str) -> str:
    """Return the skeleton for a design type with the module name substituted."""
    skeleton = RTL_SKELETONS.get(design_type, "")
    if not skeleton:
        return ""
    return skeleton.replace("<name>", design_name)


def has_ready_valid_ports(port_interface: list) -> bool:
    """Check if the port interface includes ready/valid handshake signals."""
    if not port_interface:
        return False
    names = {p.get("name", "").lower() for p in port_interface}
    return bool(names & {"s_valid", "m_valid", "s_ready", "m_ready", "valid", "ready"})


def build_protocol_assertions(port_interface: list, design_name: str) -> str:
    """Generate Verilog-2001 compatible protocol assertions for ready/valid interfaces.
    Uses manual history registers instead of $past() which requires SystemVerilog.
    Only called when has_ready_valid_ports() is True.
    """
    return f"""\
// Protocol assertions (Verilog-2001, no $past needed)
// Checks: once m_valid asserts, m_data must not change until m_ready accepts
reg [7:0] _prev_m_data;
reg       _prev_m_valid;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin _prev_m_data <= 0; _prev_m_valid <= 0; end
    else begin
        if (_prev_m_valid && !m_ready && m_valid)
            if (m_data !== _prev_m_data)
                $display("ASSERTION FAILED: m_data changed while m_valid=1 and m_ready=0 (data stability violation)");
        _prev_m_data  <= m_data;
        _prev_m_valid <= m_valid;
    end
end"""


# ==============================================================================
# AGENT: RTL GENERATOR
# ==============================================================================
def generate_rtl(
    prompt:          str,
    provider:        str,
    design_name:     str,
    architecture:    dict,
    clarified_spec:  dict  = None,
    previous_ports:  str   = None,
    previous_error:  str   = None,
    best_v_code:     str   = None,
    cand_idx:        int   = 0,
) -> dict:
    # Only cache fresh (non-mutation) calls. Mutations are intentionally unique —
    # caching them defeats the entire purpose of the evolution engine and causes
    # the generator to loop on identical candidates after the first failure.
    is_mutation = bool(previous_error and best_v_code)
    cache_key   = get_cache_key(prompt, architecture, cand_idx) if not is_mutation else None
    if cache_key and cache_key in RTL_CACHE:
        print(f"      ⚡ Cache Hit: Reusing RTL for Thread {cand_idx+1}.")
        return RTL_CACHE.get(cache_key)

    port_rule = (
        f"2. STRICT INTERFACE LOCK: You MUST reuse this exact port list verbatim:\n"
        f"   {previous_ports}"
        if previous_ports else
        "2. STRICT INTERFACE: Port declarations MUST perfectly match the Architect's `port_interface` array."
    )

    error_context = ""
    if previous_error and best_v_code:
        failure_type = classify_failure(previous_error)
        delta        = build_delta_mutation(best_v_code, failure_type, previous_error)
        delta_str    = f"\n{delta}\n" if delta else ""

        error_context = f"""
MUTATION REQUIRED — previous RTL failed.
FAILURE CLASSIFICATION: {failure_type}
SIMULATION OUTPUT:
{previous_error}
{delta_str}
TARGETED FIX RULES based on failure type:
- SHIFT_LOGIC            → check concatenation direction: left shift LSB-in is
                           out <= {{out[6:0], serial_in}}; right shift MSB-in is
                           out <= {{serial_in, out[7:1]}}; verify matches spec.
- INCREMENT_LOGIC        → modify only the increment counting path
- DECREMENT_LOGIC        → modify only the decrement counting path
- OFF_BY_ONE             → check boundary conditions (< vs <=) and initialization
- OVERFLOW_FLAG          → fix only the overflow flag assign statement
- UNDERFLOW_FLAG         → fix only the underflow flag assign statement
- RESET_LOGIC            → fix only the reset branch in always @(posedge clk or negedge rst_n)
- LOAD_LOGIC             → fix only the load_en priority branch
- PRIORITY_ORDER_BUG     → reorder if/else statements to match spec priority
- PROTOCOL_ASSERTION_FAILED → a handshake invariant was violated. Check:
                           (1) valid never drops without ready having been seen;
                           (2) s_ready deasserts when buffer is occupied AND m_ready=0;
                           (3) data does not mutate while valid is high and ready is low.
                           Fix the specific handshake state register that drives s_ready.
- LOGIC_MISMATCH         → read the expected vs got values carefully.
                           If the failing signal is identified in the log, trace it back
                           to its driver in the RTL and fix only that assignment.
                           For ready/valid pipeline designs where the RTL contains signals
                           named s_ready, m_ready, s_valid, m_valid: verify s_ready is
                           driven as: assign s_ready = !buffer_valid || m_ready;
                           (NOT assign s_ready = m_ready — that omits the buffered case)
                           Only apply this rule if those signal names are present.
- X_PROPAGATION_BUG      → explicitly initialize the signal in the reset block
- COMBINATIONAL_LOOP     → break the cyclic combinational dependency
- TB_TIMEOUT             → check for combinational loops or missing $finish paths
- TB_STRUCTURAL          → fix RTL defensively (clean interface, all outputs driven)

You MUST ONLY modify logic related to the failure type. Do not touch unrelated blocks.
PREVIOUS BEST VERILOG (mutate this, do not rewrite from scratch):
{best_v_code}
"""
    elif previous_error:
        error_context = (
            f"SYNTAX ERROR IN PREVIOUS ATTEMPT:\n{previous_error}\n"
            "Fix only the line causing this error. Do not restructure the module."
        )

    # Skeleton injection: for known design types, provide a verified-correct
    # structural starting point. The LLM fills in parameters and signal names
    # rather than inventing the architecture from scratch.
    design_type = detect_design_type(clarified_spec or {}, prompt)
    skeleton    = get_rtl_skeleton(design_type, design_name)
    skeleton_section = ""
    if skeleton and not is_mutation:
        skeleton_section = f"""
STRUCTURAL SKELETON — you MUST use this as your starting point.
Preserve the module structure, port order, always block sensitivity lists,
and handshake logic exactly. Only adapt signal widths and names to match
the spec and port_interface above. Do NOT restructure or simplify.

{skeleton}

"""
    elif skeleton and is_mutation:
        # On mutations, remind about correct structure without full skeleton
        skeleton_section = f"""
REFERENCE SKELETON for {design_type} (correct structural pattern):
{skeleton}
Compare your mutation against this reference to ensure the core structure is preserved.

"""
    alg_sections = []

    if architecture.get("control_logic"):
        lines = [
            f"  Register '{cl.get('register')}': {cl.get('algorithm')}"
            for cl in architecture["control_logic"]
        ]
        alg_sections.append(
            "REGISTER NEXT-STATE ALGORITHMS (transcribe these exactly):\n" + "\n".join(lines)
        )

    if architecture.get("flag_logic"):
        lines = [
            f"  Signal '{fl.get('signal')}' [{fl.get('type','combinational')}]: "
            f"assert when {fl.get('condition')}"
            for fl in architecture["flag_logic"]
        ]
        alg_sections.append(
            "FLAG LOGIC (combinational → assign; sequential → register):\n" + "\n".join(lines)
        )

    if architecture.get("priority_order"):
        alg_sections.append(
            "CONTROL PRIORITY (implement as nested if-else, highest first):\n"
            + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(architecture["priority_order"]))
        )

    if architecture.get("datapath_nodes"):
        lines = [
            f"  Node '{n.get('dst')}': {n.get('a')} {n.get('op')} {n.get('b')}"
            for n in architecture["datapath_nodes"]
        ]
        alg_sections.append(
            "DATAPATH NODES (implement as continuous assigns or combinational blocks):\n"
            + "\n".join(lines)
        )

    if architecture.get("muxes"):
        lines = []
        for m in architecture["muxes"]:
            lines.append(f"  MUX Target '{m.get('dst', '?')}':")
            for cond in m.get("control_conditions", []):
                if "default" in cond:
                    lines.append(f"    else → {cond['default']}")
                else:
                    lines.append(f"    if ({cond.get('when')}) → {cond.get('route')}")
        alg_sections.append(
            "MULTIPLEXERS & ROUTING (implement as prioritized if/else):\n" + "\n".join(lines)
        )

    if architecture.get("memory_blocks"):
        alg_sections.append(
            "MEMORY ARCHITECTURE (implement RAM/FIFO using provided pointers):\n  "
            + str(architecture["memory_blocks"])
        )

    if architecture.get("module_class") == "FSM":
        alg_sections.append(
            "FSM STRUCTURE: Use case(state) for state transitions. "
            "Cleanly separate state_reg and next_state."
        )

    algorithm_section = ""
    if alg_sections:
        algorithm_section = (
            "\nALGORITHM DESCRIPTION — implement by transcription, no invention:\n"
            + "\n\n".join(alg_sections) + "\n"
        )

    rtl_salt = random.randint(10000, 99999)

    system_prompt = f"""
You are an elite RTL design engineer. Generate ONLY the synthesizable Verilog RTL module.
Do NOT generate a testbench — a separate agent handles that.
{error_context}{skeleton_section}{algorithm_section}
Implement exactly this architecture:
{json.dumps(architecture, indent=2)}

SILICON RULES — all mandatory:
1. Top module named exactly: `{design_name}`
{port_rule}
3. Verilog-2001 only. No SystemVerilog. No initial blocks in synthesizable modules.
4. Sequential always @(posedge clk): non-blocking (<=) only.
   Combinational always @(*): blocking (=) only.
5. No latches: assign defaults to all signals at top of every always @(*) block.
6. Parameters: module {design_name} #(parameter WIDTH=8) (input clk, ...);
7. No expression part-selects: use intermediate wire instead of (a-b)[3:0].
8. ASYNC RESET: active-low means reset when signal is LOW.
   Sensitivity: always @(posedge clk or negedge rst_n)
   Check:       if (!rst_n) begin ... end else begin ... end
9. SINGLE DRIVER: Every reg driven by exactly ONE always block.
10. COMBINATIONAL FLAGS: If the spec says a flag "goes high WHEN condition",
    implement as: assign flag = (condition); — wire, not reg.

[VARIATION SALT: {rtl_salt}]

Return pure JSON:
{{
  "structural_reasoning": "...",
  "verilog_code": "..."
}}
"""
    result = run_llm(prompt, system_prompt, provider, max_api_retries=2, cand_idx=cand_idx)
    if result and result.get("verilog_code") and cache_key:
        RTL_CACHE.put(cache_key, result)
    return result

# ==============================================================================
# AGENT: TESTBENCH GENERATOR (black-box, receives port interface not RTL)
# ==============================================================================
def generate_testbench(
    prompt:         str,
    provider:       str,
    design_name:    str,
    port_interface: list,   # architecture["port_interface"] — NOT the RTL code
    previous_error: str = None,
    previous_tb:    str = None,
    cand_idx:       int = 0,
) -> str:
    port_str  = json.dumps(port_interface, indent=2)
    cache_key = get_cache_key(prompt, port_str, previous_error, previous_tb, cand_idx)
    if cache_key in TB_CACHE:
        print(f"      ⚡ Cache Hit: Reusing Testbench for Thread {cand_idx+1}.")
        return TB_CACHE.get(cache_key)

    error_context = ""
    if previous_error and previous_tb:
        failure_type  = classify_failure(previous_error)
        error_context = f"""
PREVIOUS TESTBENCH FAILED.
FAILURE CLASSIFICATION: {failure_type}
SIMULATION OUTPUT:
{previous_error}
FAILING TESTBENCH:
{previous_tb}
Fix only what caused the failure. Do not change passing test cases.
"""

    # Protocol assertion injection — Verilog-2001 compatible, no $past needed
    needs_protocol_assertions = has_ready_valid_ports(port_interface)
    protocol_rule = """
13. PROTOCOL ASSERTIONS (required for this ready/valid design):
    Add a manual history-register check for data stability under backpressure.
    Declare at top of module: reg [7:0] _prev_m_data; reg _prev_m_valid;
    Add this always block (Verilog-2001, no SystemVerilog):
      always @(posedge clk or negedge rst_n) begin
          if (!rst_n) begin _prev_m_data <= 0; _prev_m_valid <= 0; end
          else begin
              if (_prev_m_valid && !m_ready && m_valid && m_data !== _prev_m_data)
                  $display("ASSERTION FAILED: m_data mutated while m_valid=1 m_ready=0");
              _prev_m_data <= m_data; _prev_m_valid <= m_valid;
          end
      end
""" if needs_protocol_assertions else ""

    system_prompt = f"""
You are a hardware verification engineer. Write a Verilog-2001 testbench.
You MUST write a black-box testbench based purely on the spec and port interface.
You do NOT have access to the RTL implementation.

ORIGINAL SPEC:
{prompt}

PORT INTERFACE (instantiate the DUT using exactly these signals):
{port_str}

TESTBENCH RULES — all mandatory:
1. `timescale 1ns/1ps
2. Verilog-2001 only. No SystemVerilog anywhere.
3. Declare ALL variables at top of module, never inside begin/end blocks.
4. Drive DUT inputs through declared reg variables only.
5. Clock: always #5 clk = ~clk; with initial clk = 0;
   Period = 10ns. negedge at t=5,15,25,... posedge at t=10,20,30,...

6. CRITICAL TIMING RULE:
   The DUT registers inputs on POSEDGE clk. Outputs are valid AFTER posedge.
   You MUST wait ONE full posedge between applying inputs and reading outputs.

   MANDATORY PATTERN:
     @(negedge clk);          // apply inputs here (halfway before posedge)
     @(negedge clk);          // posedge at t+5 registered inputs; NOW read outputs
     if (out !== expected) ...  // check here, after posedge has settled

   WRONG — never do this:
     @(negedge clk); apply_inputs;
     @(posedge clk); check_outputs;  // RACE: posedge registering right now

   WRONG — never do this either:
     apply_inputs; #1; check_outputs;  // #1 does not guarantee register updated

   CORRECT WORKED EXAMPLE for a shift register (shifting in 1 then 0):
     // After reset: parallel_out = 8'b0
     @(negedge clk); shift_en=1; serial_in=1;  // apply at negedge t=5
     @(negedge clk);                             // posedge t=10 registered it
     // parallel_out is now 8'b00000001
     if (parallel_out !== 8'h01) $display("FAIL: expected 01 got %02h", parallel_out);
     @(negedge clk); shift_en=1; serial_in=0;  // apply at negedge t=15
     @(negedge clk);                             // posedge t=20 registered it
     // parallel_out is now 8'b00000010
     if (parallel_out !== 8'h02) $display("FAIL: expected 02 got %02h", parallel_out);

7. NO WHILE LOOPS. Use only bounded repeat() or for loops with fixed count.
8. GOLDEN MODEL: Implement ONLY what the spec explicitly states.
   Do not add wrap-around, saturation, or any behavior not in the spec.
   Combinational outputs (flags): check at sample negedge, same cycle as other outputs.
9. EVERY code path must call $finish.
10. On pass: $display("SIMULATION_SUCCESS"); $finish;
11. Timeout: initial begin #100000; $display("FAIL: Timeout"); $finish; end
12. COVERAGE BINS: When edge cases occur, emit:
    $display("COVER_HIT: <scenario_name>");
    e.g. COVER_HIT:overflow, COVER_HIT:reset_during_shift, COVER_HIT:shift_en_low
{protocol_rule}{error_context}

Return pure JSON:
{{
  "testbench_code": "..."
}}
"""
    result  = run_llm(prompt, system_prompt, provider, max_api_retries=2, cand_idx=cand_idx)
    tb_code = clean_code_string(result.get("testbench_code", ""))

    # Post-process: inject Verilog-2001 protocol assertions before endmodule
    # for ready/valid designs, if the LLM didn't already include them.
    if tb_code and needs_protocol_assertions and "ASSERTION FAILED" not in tb_code:
        assertion_block = build_protocol_assertions(port_interface, design_name)
        tb_code = tb_code.rstrip()
        if tb_code.endswith("endmodule"):
            tb_code = tb_code[:-len("endmodule")].rstrip()
            tb_code = tb_code + "\n\n" + assertion_block + "\n\nendmodule"

    # Structural validation — reject before it reaches Icarus and wastes a round-trip
    if not tb_code:
        return ""
    stripped = tb_code.strip()
    if not re.search(r'\bmodule\b', stripped):
        print(f"      ⚠️  TB {cand_idx+1}: Missing module declaration — discarding.")
        return ""
    if not re.search(r'\bendmodule\b', stripped):
        print(f"      ⚠️  TB {cand_idx+1}: Missing endmodule — discarding.")
        return ""
    if not re.search(r'`timescale', stripped):
        # Inject timescale rather than discard — most common omission
        tb_code = "`timescale 1ns/1ps\n" + tb_code
    # Strip any stray markdown fences that slipped through clean_code_string
    tb_code = re.sub(r'```[a-z]*\n?', '', tb_code).strip()

    if tb_code:
        TB_CACHE.put(cache_key, tb_code)
    return tb_code

# ==============================================================================
# PORT MATCHING
# ==============================================================================
def ports_match(v_code: str, locked_ports_json: str) -> bool:
    if not locked_ports_json:
        return True
    locked_ports = json.loads(locked_ports_json)

    # Handles both inline and multiline module declarations:
    #   module foo (input clk, ...);
    #   module foo\n(\n  input clk,\n  ...\n);
    m = re.search(r'module\s+\w+.*?\((.*?)\)\s*;', v_code, re.S)
    if not m:
        return False

    port_str     = m.group(1)
    declarations = re.findall(r'(?:input|output)\s[^,;]+', port_str)

    for p in locked_ports:
        name      = str(p.get("name", ""))
        direction = str(p.get("direction", ""))
        found = any(
            re.search(rf'\b{direction}\b', decl) and re.search(rf'\b{name}\b', decl)
            for decl in declarations
        )
        if not found:
            return False
    return True

# ==============================================================================
# WORKSPACE + VERIFICATION
# ==============================================================================
def save_to_workspace(design_name: str, v_code: str, tb_code: str, candidate_id: int):
    workspace = f"./workspace/{design_name}/candidate_{candidate_id}"
    os.makedirs(f"{workspace}/src", exist_ok=True)
    os.makedirs(f"{workspace}/tb",  exist_ok=True)

    v_file  = f"{workspace}/src/{design_name}.v"
    tb_file = f"{workspace}/tb/{design_name}_tb.v"

    with open(v_file,  "w") as f: f.write(v_code)
    with open(tb_file, "w") as f: f.write(tb_code)
    return v_file, tb_file

def verify_with_verilator(v_file: str, design_name: str, v_code: str):
    if not re.search(r'\bmodule\s+' + re.escape(design_name) + r'\b', v_code):
        return False, f"%Error: Top module '{design_name}' not found."
    cmd = ["verilator", "--lint-only", "-Wall",
           "-Wno-DECLFILENAME", "-Wno-UNUSED", "-Wno-UNDRIVEN",
           "-Werror-WIDTH", "-Werror-IMPLICIT", v_file]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if "UNOPTFLAT" in r.stderr:
            return False, "COMBINATIONAL LOOP DETECTED (UNOPTFLAT)."
        if r.returncode == 0:
            return True, "Clean"
        return False, truncate_log(r.stderr)
    except subprocess.TimeoutExpired:
        return False, "Verilator timeout."

def verify_with_iverilog(v_file: str, tb_file: str, design_name: str, candidate_id: int):
    workspace  = f"./workspace/{design_name}/candidate_{candidate_id}"
    output_bin = f"{workspace}/{design_name}_sim.vvp"

    cr = subprocess.run(
        ["iverilog", "-o", output_bin, v_file, tb_file],
        capture_output=True, text=True
    )
    if cr.returncode != 0:
        return False, f"Compile failed:\n{truncate_log(cr.stderr)}", 0

    try:
        sr = subprocess.run(["vvp", output_bin],
                            capture_output=True, text=True, timeout=60)
        cover_hits = len(set(re.findall(r'COVER_HIT:\s*(\w+)', sr.stdout)))

        if sr.returncode != 0:
            return False, f"Runtime failed:\n{truncate_log(sr.stderr)}", cover_hits
        if any(kw in sr.stdout for kw in
               ("ERROR", "Error", "FAIL", "Fail", "ASSERTION FAILED")):
            return False, f"Verification failed:\n{truncate_log(sr.stdout, 15)}", cover_hits
        if "SIMULATION_SUCCESS" not in sr.stdout:
            return False, "Missing SIMULATION_SUCCESS — simulation exited prematurely.", cover_hits
        return True, "Simulation clean.", cover_hits
    except subprocess.TimeoutExpired:
        return False, "Simulation timeout (possible infinite loop).", 0

def verify_with_yosys(v_file: str, base_prompt: str):
    cmd = ["yosys", "-p",
           f"read_verilog {v_file}; proc; opt; fsm; memory; check -assert; prep; stat"]
    try:
        r   = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        log = r.stderr + "\n" + r.stdout
        if r.returncode == 0 and "ERROR" not in log:
            dffs  = len(re.findall(r'\$(?:a?dff|dffe|sdff)', log, re.IGNORECASE))
            cells = re.search(r'Number of cells:\s+(\d+)', log)
            seq_kw = ["clk", "clock", "sync", "sequential", "posedge", "negedge"]
            if dffs == 0 and any(kw in base_prompt.lower() for kw in seq_kw):
                return False, "CRITICAL: Sequential design expected but 0 registers synthesized."
            return True, f"Cells: {cells.group(1) if cells else '?'}, DFFs: {dffs}"
        return False, truncate_log(log, 20)
    except subprocess.TimeoutExpired:
        return False, "Yosys timeout."

# ==============================================================================
# MAIN EVOLUTIONARY BUILD LOOP
# ==============================================================================
def autonomous_build_loop(
    base_prompt:          str,
    design_name:          str,
    provider:             str,
    max_retries:          int = 10,
    candidates_per_round: int = 3,
) -> bool:

    print(f"\n🚀 ChipGPT | Design: '{design_name}' | Provider: {provider.upper()}")
    print(f"   🧠 Fast model  : {FAST_MODEL}")
    print(f"   🔥 Strong model: {STRONG_MODEL}\n")

    clarified_spec = generate_specification(base_prompt, provider)
    if not clarified_spec:
        print("   ❌ Clarifier returned empty spec.")
        return False

    arch_hashes          = set()
    code_hashes          = set()
    locked_ports         = None
    arch_failures        = {}   # arch_hash → set of failure_type strings
    error_log            = None
    architecture         = None
    architecture_flawed  = True
    invalid_arch_count   = 0
    duplicate_arch_count = 0

    # Tuple scoring: (pipeline_stage, secondary_quality)
    # stage: 0=no lint pass, 1=lint, 2=sim, 3=synth, 4=gate-level
    best_score           = (-1, -999999)
    best_v_code          = None
    best_tb_code         = None
    best_err_log         = None
    global_candidate_id  = 0
    consecutive_sim_fails = 0
    last_err_fingerprint  = None
    same_err_count        = 0

    for attempt in range(max_retries):
        print(f"\n{'─'*60}")
        print(f"🔄 Generation {attempt + 1}/{max_retries}")
        print(f"{'─'*60}")

        # ── Architecture redesign triggers ────────────────────────────────────
        trigger_redesign = False
        if consecutive_sim_fails >= 5:
            trigger_redesign = True
            print("   🔄 5 consecutive sim failures — forcing full architecture redesign...")
        elif architecture:
            current_arch_hash = code_hash(json.dumps(architecture, sort_keys=True))
            if len(arch_failures.get(current_arch_hash, set())) >= 3:
                trigger_redesign = True
                print("   🔄 3 distinct failure types in this architecture — redesigning...")

        if trigger_redesign:
            architecture_flawed   = True
            consecutive_sim_fails = 0
            best_score            = (-1, -999999)
            best_v_code           = None
            best_tb_code          = None
            locked_ports          = None
            fail_history_str = "\n".join(
                f"- Arch {h[:6]}: " + ", ".join(reasons)
                for h, reasons in arch_failures.items()
            )
            error_log = (
                f"CRITICAL: Previous architectures failed.\n"
                f"Failure History:\n{fail_history_str}\n"
                f"Produce a structurally DIFFERENT blueprint."
            )

        if architecture_flawed:
            enforced_style = ARCH_STYLES[attempt % len(ARCH_STYLES)]
            architecture   = generate_architecture(
                clarified_spec, provider, error_log, attempt, enforced_style
            )
            arch_hash = code_hash(json.dumps(architecture, sort_keys=True))

            if not validate_architecture(architecture):
                invalid_arch_count += 1
                if invalid_arch_count > 5:
                    print("   ❌ FATAL: Architect stuck in invalid schema loop.")
                    return False
                print(f"   ⚠️  Architect returned invalid blueprint. Retrying...")
                error_log = (
                    "CRITICAL: Blueprint missing required fields. "
                    "Ensure port_interface has a clk port and registers are defined."
                )
                continue

            if arch_hash in arch_hashes:
                duplicate_arch_count += 1
                if duplicate_arch_count > 5:
                    print("   ⚠️  Architect stuck in duplicate loop. Forcing acceptance.")
                    architecture_flawed  = False
                    duplicate_arch_count = 0
                else:
                    print("   ⚠️  Duplicate blueprint. Forcing variation...")
                    error_log = "CRITICAL: Do not repeat previous blueprints. Change structure."
                    continue
            else:
                arch_hashes.add(arch_hash)
                architecture_flawed  = False
                duplicate_arch_count = 0
                invalid_arch_count   = 0
                print(f"   📐 Architect: New blueprint (class={architecture.get('module_class','?')}, style={enforced_style})")
        else:
            print("   📐 Architect: Reusing stable blueprint.")

        # ── RTL generation (parallel) ─────────────────────────────────────────
        if best_v_code:
            print(f"   🧬 Evolution Engine: Mutating best RTL (score=Stage {best_score[0]})...")
        else:
            print(f"   🛠️  Generator: Spawning {candidates_per_round} parallel RTL candidates...")

        rtl_candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidates_per_round) as executor:
            futures = {}
            for i in range(candidates_per_round):
                time.sleep(2)
                dynamic_idx = i + same_err_count  # shift temperature on repeated errors
                futures[executor.submit(
                    generate_rtl,
                    base_prompt, provider, design_name, architecture,
                    clarified_spec,
                    locked_ports, best_err_log or error_log, best_v_code,
                    dynamic_idx
                )] = i

            try:
                for future in concurrent.futures.as_completed(futures, timeout=420):
                    try:
                        result = future.result(timeout=200)
                    except Exception:
                        continue

                    v_code = clean_code_string(result.get("verilog_code", ""))
                    if len(v_code) < 50:
                        continue
                    if locked_ports and not ports_match(v_code, locked_ports):
                        print("   ⚠️  Port interface mismatch — candidate discarded.")
                        continue

                    h = code_hash(v_code)
                    if h in code_hashes:
                        print("   ⚠️  Duplicate candidate — discarded.")
                        continue
                    code_hashes.add(h)

                    has_memory_blocks = bool(architecture.get("memory_blocks"))
                    if (architecture.get("module_class") == "DATAPATH"
                            and has_memory_blocks
                            and not re.search(r'reg\s*\[[^\]]+\]\s*\w+\s*\[[^\]]+\]', v_code)):
                        print("   ⚠️  DATAPATH with memory_blocks missing 2D reg. Discarding.")
                        continue

                    rtl_candidates.append(v_code)

            except concurrent.futures.TimeoutError:
                print("   ⚠️  RTL executor timeout. Restarting round.")

        if not rtl_candidates:
            print("   ❌ No valid RTL candidates this round.")
            error_log = "All RTL candidates were empty, had wrong ports, or were duplicates."
            continue

        # ── Reviewer pass ─────────────────────────────────────────────────────
        print(f"   🔎 Reviewer: Scanning {len(rtl_candidates)} RTL candidate(s)...")
        reviewed_candidates = []
        for idx, v_code in enumerate(rtl_candidates):
            review = review_hardware(v_code, provider)
            if review.get("status") == "REJECTED":
                fixed = clean_code_string(review.get("fixed_code", ""))
                if (fixed
                        and "module " in fixed
                        and "endmodule" in fixed
                        and design_name in fixed):
                    has_assign_before = bool(re.search(r'\bassign\b', v_code))
                    has_assign_after  = bool(re.search(r'\bassign\b', fixed))
                    always_before     = len(re.findall(r'\balways\b', v_code))
                    always_after      = len(re.findall(r'\balways\b', fixed))
                    if has_assign_before and not has_assign_after:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer removed assigns. Keeping original.")
                        reviewed_candidates.append(v_code)
                    elif always_after < always_before:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer removed always block. Keeping original.")
                        reviewed_candidates.append(v_code)
                    elif 0.75 < len(fixed) / len(v_code) < 1.3:
                        print(f"      ✅ Candidate {idx+1}: Patch applied.")
                        reviewed_candidates.append(fixed)
                    else:
                        print(f"      ⚠️  Candidate {idx+1}: Hallucinated rewrite. Keeping original.")
                        reviewed_candidates.append(v_code)
                else:
                    print(f"      ⚠️  Candidate {idx+1}: Reviewer broke module structure. Keeping original.")
                    reviewed_candidates.append(v_code)
            else:
                reviewed_candidates.append(v_code)

        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture["port_interface"])
            print(f"   🔒 Port interface locked ({len(architecture['port_interface'])} ports).")

        # ── Testbench generation ──────────────────────────────────────────────
        _structural_tb_failures = {
            "TB_TIMEOUT", "SYNTAX_ERROR", "TB_STRUCTURAL", "COMBINATIONAL_LOOP"
        }
        tb_failure_type   = classify_failure(best_err_log) if best_err_log else None
        tb_is_logic_failure = (
            best_tb_code is not None
            and tb_failure_type is not None
            and tb_failure_type not in _structural_tb_failures
        )

        port_interface = architecture.get("port_interface", [])

        print(f"   📋 Testbench Agent: ", end="")
        if tb_is_logic_failure:
            print("Reusing locked testbench (logic failure — TB is correct fitness function).")
        else:
            print(f"Generating {len(reviewed_candidates)} testbench(es) in parallel...")

        def _gen_tb(args):
            idx, v_code = args
            if tb_is_logic_failure:
                return idx, v_code, best_tb_code
            # Only pass previous TB error context when the previous TB was structurally
            # broken (syntax error, timeout). For logic failures the TB is locked above.
            # Passing a logic-failure error as "fix the testbench" context to a fresh
            # generation would tell the model the RTL is correct and corrupt a good TB.
            tb_error = best_err_log if (
                best_tb_code is not None
                and tb_failure_type in _structural_tb_failures
            ) else None
            prev_tb = best_tb_code if tb_error else None
            tb = generate_testbench(
                base_prompt, provider, design_name,
                port_interface,
                tb_error,
                prev_tb,
                idx,
            )
            return idx, v_code, tb

        candidates = []
        tb_workers = max(1, min(2, len(reviewed_candidates)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=tb_workers) as tb_executor:
            tb_futures = [
                tb_executor.submit(_gen_tb, (idx, v_code))
                for idx, v_code in enumerate(reviewed_candidates)
            ]
            try:
                for tb_future in concurrent.futures.as_completed(tb_futures, timeout=300):
                    try:
                        idx, v_code, tb_code = tb_future.result(timeout=200)
                    except Exception as e:
                        print(f"      ⚠️  Testbench generation failed: {e}")
                        continue
                    if not tb_code or len(tb_code) < 50:
                        print(f"      ⚠️  Testbench {idx+1}: Empty. Discarding.")
                        continue
                    status = "🔒 Locked" if tb_is_logic_failure else "✅ Generated"
                    print(f"      {status} Testbench {idx+1}.")
                    candidates.append((v_code, tb_code))
            except concurrent.futures.TimeoutError:
                for f in tb_futures:
                    f.cancel()
                print(f"   ⚠️  Testbench timeout — {len(candidates)} collected before cutoff.")

        if not candidates:
            print("   ❌ No valid candidates after testbench generation.")
            continue

        # ── Verification ──────────────────────────────────────────────────────
        round_passed     = False
        round_sim_passed = False

        for v_code, tb_code in candidates:
            global_candidate_id += 1
            print(f"\n   🔬 Testing Candidate #{global_candidate_id}...")

            v_file, tb_file  = save_to_workspace(design_name, v_code, tb_code, global_candidate_id)
            current_score    = (0, 0)

            passed, log = verify_with_verilator(v_file, design_name, v_code)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Verilator: {err}")
                current_score = (0, -len(err))
                if current_score > best_score:
                    best_score   = current_score
                    best_err_log = f"Syntax/Lint Error:\n{err}"
                continue

            print("      ✅ Verilator passed.")
            passed, log, cover_hits = verify_with_iverilog(
                v_file, tb_file, design_name, global_candidate_id
            )
            if not passed:
                err              = extract_primary_error(log)
                full_sim_context = truncate_log(log, 30)
                failure_type     = classify_failure(full_sim_context)
                print(f"      ❌ Icarus [{failure_type}]: {err}")

                # Same-error detection → unlock TB if stuck
                err_fp = get_error_fingerprint(failure_type, full_sim_context)
                if err_fp == last_err_fingerprint:
                    same_err_count += 1
                else:
                    last_err_fingerprint = err_fp
                    same_err_count       = 1

                if same_err_count >= 4:
                    print(f"      🔁 Same error repeated {same_err_count}x — unlocking TB.")
                    best_tb_code   = None
                    same_err_count = 0

                mismatches    = len(re.findall(
                    r'(fail|error|mismatch|expected|assert)', log, re.IGNORECASE
                ))
                current_score = (1, (cover_hits * 10) - mismatches)

                if current_score > best_score:
                    best_score   = current_score
                    best_v_code  = v_code
                    best_tb_code = tb_code

                    # Extract the specific FAIL assertion lines — these are the most
                    # actionable feedback for the mutation engine, more useful than
                    # the full truncated log which buries them.
                    fail_lines = "\n".join(
                        line for line in log.splitlines()
                        if re.search(r'\bFAIL\b|\bfail\b|expected|mismatch|assertion', line, re.I)
                    )
                    fail_summary = (
                        f"SPECIFIC FAILING ASSERTIONS:\n{fail_lines}\n\n"
                        if fail_lines else ""
                    )

                    best_err_log = (
                        f"FAILURE TYPE: {failure_type}\n"
                        f"{fail_summary}"
                        f"Simulation Error (full output):\n{full_sim_context}"
                    )

                if architecture:
                    ah = code_hash(json.dumps(architecture, sort_keys=True))
                    if ah not in arch_failures:
                        arch_failures[ah] = set()
                    arch_failures[ah].add(failure_type)
                continue

            round_sim_passed = True
            print(f"      ✅ Icarus simulation passed. (Coverage Hits: {cover_hits})")

            passed, log = verify_with_yosys(v_file, base_prompt)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Yosys: {err}")
                current_score = (2, 0)
                if current_score > best_score:
                    best_score   = current_score
                    best_v_code  = v_code
                    best_tb_code = tb_code
                    best_err_log = f"Synthesis Error:\n{err}"
                continue

            print(f"      ✅ Yosys passed: {log}")
            current_score = (3, 0)
            if current_score > best_score:
                best_score   = current_score
                best_v_code  = v_code
                best_tb_code = tb_code

            round_passed = True
            break

        if round_passed:
            print(f"\n{'═'*60}")
            print("🎉 PIPELINE COMPLETE — Structural + Functional + Synthesis verification passed.")
            print(f"📁 Winning design: ./workspace/{design_name}/candidate_{global_candidate_id}/")
            print(f"{'═'*60}\n")
            return True

        if best_score[0] <= 1 and not round_sim_passed:
            consecutive_sim_fails += 1
        else:
            consecutive_sim_fails = 0

        print(f"\n   📊 Round summary: best_score=Stage {best_score[0]} | "
              f"sim_fails_streak={consecutive_sim_fails} | "
              f"candidates_tested={global_candidate_id}")

    print("\n🚨 Max retries reached without a passing design.")
    return False

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChipGPT: Autonomous RTL Generator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prompt",       type=str)
    parser.add_argument("--name",       type=str, default="my_module")
    parser.add_argument("--provider",   type=str, choices=["openai", "groq"], default="openai")
    parser.add_argument("--retries",    type=int, default=10)
    parser.add_argument("--candidates", type=int, default=3)
    args = parser.parse_args()

    success = autonomous_build_loop(
        base_prompt=args.prompt,
        design_name=args.name,
        provider=args.provider,
        max_retries=args.retries,
        candidates_per_round=args.candidates,
    )
    exit(0 if success else 1)
