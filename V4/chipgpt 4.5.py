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

# External skeleton library — versioned, testable independently
from skeletons import (
    detect_design_type,
    get_skeleton,
    check_skeleton_invariants,
    has_ready_valid_ports,
    build_protocol_assertions,
    SKELETON_LIBRARY,
)

# ==============================================================================
# MODEL TIER CONFIGURATION
# ==============================================================================
FAST_MODEL   = "gpt-4o-mini"
STRONG_MODEL = "gpt-5-mini"

REASONING_MODELS = {
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "gpt-5-mini", "gpt-5",
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
    # Strip COVER_HIT lines — they are coverage markers, not failure descriptions.
    # Without this, "COVER_HIT:reset_during_shift" would trigger SHIFT_LOGIC
    # even when the actual failure is a reset bug.
    stripped = "\n".join(
        line for line in log.splitlines()
        if not line.strip().upper().startswith("COVER_HIT")
    )
    l = stripped.lower()
    if "timeout" in l or "simulation timeout" in l:
        return "TB_TIMEOUT"
    if "unoptflat" in l or ("loop" in l and "comb" in l):
        return "COMBINATIONAL_LOOP"
    if "compile" in l or "syntax" in l:
        return "SYNTAX_ERROR"
    if "latch" in l:
        return "LATCH_INFERRED"
    if "bit extraction" in l and "requires" in l and "bit index" in l:
        return "BIT_EXTRACTION"
    if "width" in l or "size" in l or "expects" in l:
        return "WIDTH_MISMATCH"
    # FSM/UART output failures — check before X/Z regex to avoid false matches
    if any(kw in l for kw in ("tx_out", "start bit", "stop bit", "data bit",
                               "baud", "uart", "serial bit")):
        return "FSM_OUTPUT_WRONG"
    # Handshake/ready-valid failures — check before OFF_BY_ONE numeric diff
    if any(kw in l for kw in ("s_ready", "m_ready", "m_valid", "s_valid",
                               "handshake", "backpressure", "buffer", "skid")):
        return "HANDSHAKE_LOGIC"
    # X/Z propagation: tightened regex — only match clear X/Z indicators
    if re.search(r'\bgot\s+z\b|\bvalue\s*=\s*z\b', l):
        return "UNDRIVEN_OUTPUT"
    if re.search(r'\bxx+\b|\bgot\s+x\b|=\s*x\b', l):
        return "X_PROPAGATION_BUG"
    if "stuck" in l or "never changes" in l:
        return "STATE_MACHINE_STUCK"

    m = re.search(
        r'expected\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+).*?got\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+)', l
    )
    if m:
        try:
            g1, g2 = m.group(1), m.group(2)
            if re.search(r'[xzXZ]', g1 + g2):
                pass
            else:
                has_hex = ('0x' in l or any(c in 'abcdef'
                           for c in g1.lower() + g2.lower()))
                base = 16 if has_hex else 10
                exp  = int(g1, base)
                got  = int(g2, base)
                # Guard: OFF_BY_ONE only for multi-bit values (>1).
                # Single-bit 0 vs 1 differences are handshake/flag bugs.
                if abs(exp - got) == 1 and max(exp, got) > 1:
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
    # UART/FSM-specific reset failure: FSM didn't return to IDLE after async reset
    if re.search(r'tx_ready.*got\s+0|after.*reset.*tx_ready|recovery.*reset', l):
        return "RESET_AFTER_TX"
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

def auto_patch_param_width(v_code: str) -> str:
    """DISABLED: auto-widening was breaking internally-consistent LLM-generated code.
    The patcher widened register declarations but not the sized literals in assignments,
    creating internal inconsistency that Verilator rejects. The LLM now handles this
    via explicit prompt rules (unsized literals, correct counter widths).
    """
    return v_code


def clean_code_string(raw_code: str) -> str:
    if not raw_code:
        return ""
    clean = raw_code.strip()
    triple_tick = "`" * 3
    for fence in (f"{triple_tick}verilog", f"{triple_tick}systemverilog",
                  f"{triple_tick}v", triple_tick):
        clean = clean.replace(fence, "")
    return clean.strip()


def _check_blocking_mix(v_code: str) -> str:
    """Static pre-flight check for BLKANDNBLK (mixed blocking/non-blocking on same var).
    Returns the offending variable name if found, empty string if clean.
    Only checks inside always @(posedge...) blocks — blocking is legal in always @(*).
    """
    # Extract sequential always blocks only
    seq_blocks = re.findall(
        r'always\s*@\s*\(\s*posedge.*?(?=always\s*@|\Z)',
        v_code, re.DOTALL | re.IGNORECASE
    )
    for block in seq_blocks:
        nonblock_vars = set(
            m.group(1).strip().split('[')[0].strip()   # strip bit-select
            for m in re.finditer(r'(\w[\w\[\]:]*)\s*<=', block)
            if m.group(1).strip() not in ('', 'begin', 'end')
        )
        block_vars = set(
            m.group(1).strip().split('[')[0].strip()
            for m in re.finditer(r'(?<!<)(?<!=)\b(\w+)\s*=[^=<>!]', block)
            if m.group(1).strip() not in ('', 'begin', 'end', 'if', 'else',
                                           'case', 'default', 'integer', 'reg')
        )
        conflict = nonblock_vars & block_vars
        if conflict:
            return next(iter(conflict))
    return ""

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
        # Only hard-reject if BOTH fields are completely absent (empty dict and empty list)
        # A minimal FSM with just state names but no full transition table is still usable.
        se = fsm.get("state_encoding", {})
        nl = fsm.get("next_state_logic", [])
        if not se and not nl and not fsm.get("states"):
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

CRITICAL RULES — violating any of these will cause the patch to be rejected:
  - DO NOT rewrite the module. Output only a minimal patch.
  - NEVER convert assign statements to reg assignments.
  - NEVER add a new always block. Fix latches by adding defaults INSIDE existing always blocks.
  - NEVER drive a register in an always @(*) block if it is already driven in an always @(posedge clk) block.
    This creates BLKANDNBLK conflicts that make the design unsynthesizable.
  - If the code is clean, return {"status": "PASSED", "fixed_code": ""}.
  - If patching, return {"status": "REJECTED", "fixed_code": "<full corrected module>"}.

CORRECT latch fix pattern (add default at top of existing block):
  always @(*) begin
    out = 1'b0;        // ← add this default line
    if (cond) out = 1'b1;
  end

WRONG pattern (NEVER do this):
  always @(*) begin    // ← NEVER add a new always block to fix a latch
    reg_driven_by_posedge_block = 0;
  end

Return pure JSON: {"status": "...", "fixed_code": "..."}
"""
    return run_llm(v_code, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)



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
    design_type_hint: str  = None,   # passed from main loop to avoid re-detection
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

        # Design-type-specific mutation guidance — only injected when relevant.
        # For bridge/adapter designs (design_type=="generic"), also check if the
        # prompt mentions a known peripheral — inject protocol hints without forcing
        # the skeleton structure.
        _dtype = design_type  # captured from detect above
        _prompt_lower = prompt.lower()
        _dsrules = ""
        if _dtype == "uart_tx" or ("uart" in _prompt_lower and _dtype == "generic"):
            _dsrules = """UART TX PROTOCOL RULES (applies to any UART transmitter, including AXI bridges):
- FRAME COUNTER: A UART frame has 10 phases (start + 8 data + stop).
  Use a 4-bit counter reg [3:0] phase to count 0-9.
  Use a separate 3-bit index or slice when indexing the 8-bit data register.
- BAUD COUNTER: Use reg [31:0] baud_counter for any unknown TICKS_PER_BAUD parameter.
  Compare as: if (baud_counter == TICKS_PER_BAUD - 1).
- STATE TRANSITIONS: In STOP state, count FULL baud period before returning to IDLE.
  STOP→IDLE only when baud_counter == TICKS_PER_BAUD - 1 (or equivalent).
- READY SIGNAL: The "ready" output (tx_ready, s_axis_tready, or equivalent) MUST be:
  (a) set to 1 in the reset block
  (b) set to 1 when entering IDLE state
  (c) set to 0 when leaving IDLE (capturing new data)
  FORGETTING to reassert ready in the IDLE state or reset block is the #1 AXI-UART bug.
"""
        elif _dtype == "skid_buffer":
            _dsrules = """SKID BUFFER SPECIFIC RULES:
- s_ready MUST be: assign s_ready = !buf_valid || m_ready;
- Drain path priority: buf_valid check BEFORE s_valid check inside if(m_ready||!m_valid).
- Push path: write to buf_data/buf_valid ONLY — never overwrite m_data in push path.
"""

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
- RESET_AFTER_TX         → FSM did not return to IDLE after async reset.
                           Every output register and state register MUST be explicitly
                           assigned in the if (!rst_n) block. Never rely on state=0
                           or any implicit default — every reg needs an explicit reset value.
                           Check: is the output register (e.g. tx_ready, s_axis_tready,
                           or equivalent "ready" signal) assigned in the reset block?
- RESET_LOGIC            → fix only the reset branch in always @(posedge clk or negedge rst_n)
- LOAD_LOGIC             → fix only the load_en priority branch
- PRIORITY_ORDER_BUG     → reorder if/else statements to match spec priority
- PROTOCOL_ASSERTION_FAILED → a handshake invariant was violated.
                           Read the FAIL message to identify which signal is wrong.


                           "ready not deasserted immediately after data capture" / tready=1 when it should be 0:
                           ready must go LOW in the SAME clock cycle as data capture.
                           The handshake fires at posedge when tvalid=1 AND tready=1.
                           In that SAME always block, you MUST set tready<=0.
                           WRONG (ready stays high one extra cycle — second word captured):
                             if (s_axis_tvalid) begin data<=tdata; state<=START; end
                           CORRECT (ready deasserts same cycle as capture):
                             if (s_axis_tvalid && s_axis_tready) begin
                               data<=s_axis_tdata; s_axis_tready<=0; state<=START;
                             end

                           "ready not reasserted after frame" / "tready=0" after transmission:
                           The ready signal (s_axis_tready, tx_ready, etc.) was never set back
                           to 1 after the UART frame completed. Check THREE places:
                           (a) Reset block: ready must be initialized to 1 (device starts ready)
                           (b) IDLE state assignment: always set ready=1 in IDLE state or
                               when entering IDLE — never rely on it staying 1 from before.
                           (c) State transition to IDLE: the cycle that transitions STOP→IDLE
                               must also execute ready<=1 in the same clock edge.
                           Correct pattern:
                             STOP: if (baud_done) begin state<=IDLE; s_axis_tready<=1'b1; end
                           Wrong pattern (forgets ready):
                             STOP: if (baud_done) state<=IDLE;

                           "valid never drops" / "data mutated while valid":
                           (1) valid must stay asserted until ready is seen;
                           (2) data must not change while valid=1 and ready=0.
- LOGIC_MISMATCH         → read the expected vs got values carefully.
                           If the failing signal is identified in the log, trace it back
                           to its driver in the RTL and fix only that assignment.
                           For ready/valid pipeline designs where the RTL contains signals
                           named s_ready, m_ready, s_valid, m_valid: verify s_ready is
                           driven as: assign s_ready = !buffer_valid || m_ready;
                           (NOT assign s_ready = m_ready — that omits the buffered case)
                           Only apply this rule if those signal names are present.
- UNDRIVEN_OUTPUT        → an output port reads as 'z' (high-impedance) meaning it has
                           no driver. Fix: declare it as 'output reg' and add an explicit
                           assignment in the reset block and in the sequential logic.
                           Example: 'output reg m_valid' with 'm_valid <= 1\'b0' in reset.
- HANDSHAKE_LOGIC        → a ready/valid handshake signal has the wrong value.
                           THREE distinct sub-bugs — identify which matches the log:

                           SUB-BUG A: s_ready wrong
                           MANDATORY: assign s_ready = !buf_valid || m_ready;
                           Not !buf_valid alone (drops m_ready).
                           Not m_ready alone (loses buffer term).

                           SUB-BUG B: buffered data not appearing (m_valid=0 after backpressure)
                           Log: "expected buffered ... m_valid=0"
                           Drain path MUST be:
                             if (m_ready || !m_valid) begin
                               if (buf_valid) begin
                                 m_valid <= 1'b1; m_data <= buf_data; buf_valid <= 1'b0;
                               end else if (s_valid) begin
                                 m_valid <= 1'b1; m_data <= s_data;
                               end else begin m_valid <= 1'b0; end
                             end
                           Common mistake: setting m_data without m_valid<=1'b1.

                           SUB-BUG C: m_data mutated under backpressure
                           Log: "ASSERTION FAILED: m_data mutated while m_valid=1 m_ready=0"
                           Push path must write to buf_data ONLY, never m_data:
                             if (s_valid && s_ready && !(m_ready || !m_valid)) begin
                               buf_valid <= 1'b1; buf_data <= s_data;
                             end
- FSM_OUTPUT_WRONG       → a state machine output has the wrong value.
                           General approach:
                           (1) Find which state the FSM should be in when the output is wrong.
                           (2) Check that state's output assignment — is the value correct?
                           (3) Check that the FSM actually transitions into that state.
                               Missing or wrong transition condition → FSM stays in wrong state.
                           (4) Check the baud/timing counter: must count 0 to N-1 before
                               transitioning. Off-by-one means transitioning one cycle early/late.
                           (5) Check state register update: ensure state<=next_state fires
                               unconditionally in the sequential always block.
                           (6) Verify every output register has an explicit value in every
                               state — no undriven paths that rely on previous state value.
- BIT_EXTRACTION         → array index is wider than the array requires.
                           Verilator: "Bit extraction of var[N:0] requires M bit index, not K bits"
                           means your index variable is K bits but the array only needs M bits.

                           UART/SERIAL FRAME COUNTER NOTE — this is the most common cause:
                           A UART frame has 10 phases (start + 8 data + stop), so the frame
                           counter needs 4 bits (reg [3:0]) to count 0-9.
                           But the DATA array is only 8 bits, so indexing it needs 3 bits.
                           SOLUTION: Use a 4-bit frame counter but slice it when indexing data:
                             reg [3:0] phase;           // counts 0-9: start(0) data(1-8) stop(9)
                             tx_out <= tx_data[phase[2:0] - 1]; // slice for 8-bit array index
                           OR use a separate 3-bit data index alongside the 4-bit frame counter.
                           NEVER use a 3-bit counter for a 10-phase UART frame (max value 7 < 9).

                           General fixes (non-UART):
                           Option A — Narrow the counter to match array size:
                             reg [2:0] bit_counter;   // for 8-element array (indices 0-7)
                           Option B — Slice at point of use (keeps counter wide for other uses):
                             assign tx = shift_reg[bit_counter[2:0]];  // slice to 3 bits at use
                           Look at the Verilator error line — it shows exactly where the mismatch is.
- WIDTH_MISMATCH         → signal has wrong bit width for its context.
                           RULE 1 — INDEX WIDTH: To index into a reg[N-1:0] array, the index
                               variable must be exactly $clog2(N) bits wide.
                               reg [7:0] data needs a [2:0] index (3 bits, values 0-7).
                               reg [15:0] data needs a [3:0] index (4 bits, values 0-15).
                               Check the Verilator error: "requires X bit index, not Y bits"
                               means your index is Y bits but needs to be X bits.
                           RULE 2 — COMPARISON WIDTH: Both sides of == must have matching widths.
                               reg [2:0] counter == 3'd7   ✓ (both 3 bits)
                               reg [2:0] counter == 4'd7   ✗ (3 vs 4 bits — WIDTH error)
                               NEVER value-overflow: 3-bit register can hold 0-7 maximum.
                               Comparing reg[2:0] against 8 or higher is always wrong.
                               Fix: widen the register OR reduce the comparison value.
                           RULE 3 — LITERAL FORMAT: Never combine number prefixes.
                               3'd2 or 2'b10 or 2'h2 — pick ONE format per literal.
                               3'd2'b10 is illegal syntax (two prefixes concatenated).
                           RULE 4 — PARAMETER COMPARISONS: comparing against a parameter is safe
                               because Verilog auto-widens for parameter comparisons.
                               if (counter == PARAM - 1) is legal regardless of counter width.
- X_PROPAGATION_BUG      → explicitly initialize the signal in the reset block
- COMBINATIONAL_LOOP     → break the cyclic combinational dependency
- TB_TIMEOUT             → check for combinational loops or missing $finish paths
- TB_STRUCTURAL          → fix RTL defensively (clean interface, all outputs driven)

You MUST ONLY modify logic related to the failure type. Do not touch unrelated blocks.
{_dsrules}PREVIOUS BEST VERILOG (mutate this, do not rewrite from scratch):
{best_v_code}
"""
    elif previous_error:
        error_context = (
            f"SYNTAX ERROR IN PREVIOUS ATTEMPT:\n{previous_error}\n"
            "Fix only the line causing this error. Do not restructure the module."
        )

    # Skeleton injection from external library (skeletons.py)
    design_type = design_type_hint or detect_design_type(clarified_spec or {}, prompt)
    # Infer width from architecture if available
    width = 8
    for reg in architecture.get("registers", []):
        if reg.get("width"):
            try:
                width = int(reg["width"])
                break
            except (ValueError, TypeError):
                pass

    skeleton = get_skeleton(design_type, design_name, width)
    skeleton_section = ""
    if skeleton and not is_mutation:
        skeleton_section = f"""
STRUCTURAL SKELETON — you MUST use this as your starting point.
This skeleton encodes invariants that are FROZEN and must not be changed:
the sensitivity list, reset structure, non-blocking discipline, and (for
skid buffers) the s_ready equation and drain/push priority.
Only adapt signal widths, port names, and behavioral branches to match the spec.
DO NOT remove reset initializations. DO NOT introduce blocking (=) assignments
inside always @(posedge clk) blocks.

{skeleton}

"""
    elif skeleton and is_mutation:
        skeleton_section = f"""
REFERENCE SKELETON for {design_type} — compare your mutation against this.
Ensure the frozen structural patterns are preserved:
{skeleton}

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
4. Sequential always @(posedge clk): non-blocking (<=) ONLY.
   Combinational always @(*): blocking (=) ONLY.
   NEVER mix blocking and non-blocking on the same variable — BLKANDNBLK is fatal.

   WRONG (causes BLKANDNBLK — never do this):
     always @(posedge clk) begin state = next_state; end  // blocking in sequential
     always @(*) begin state <= IDLE; end                  // nonblocking in combinational
     always @(posedge clk) begin count <= count + 1; end
     always @(*) begin count = 0; end                      // second driver on count

   CORRECT (one driver per register, always in sequential block):
     always @(posedge clk or negedge rst_n) begin
         if (!rst_n) state <= IDLE;
         else        state <= next_state;
     end

4b. COUNTER AND REGISTER WIDTH RULES — mandatory for Verilator compatibility:
   When a counter is compared against a module parameter (e.g. TICKS_PER_BAUD),
   declare the counter wide enough to hold the parameter's maximum value.
   Use 32-bit counters for parameters whose size is unknown at design time:
     reg [31:0] baud_counter;   // safe for any TICKS_PER_BAUD value

   NEVER use explicitly-sized small literals for wide registers:
   WRONG:  baud_counter <= 16'd0;   // 16-bit literal into 32-bit reg → WIDTH error
   WRONG:  baud_counter + 16'd1     // 16-bit addend with 32-bit reg → WIDTH error
   CORRECT: baud_counter <= 0;       // unsized — Verilog auto-sizes to match LHS
   CORRECT: baud_counter <= baud_counter + 1;  // unsized integer constant
   CORRECT: baud_counter <= 'd0;    // explicitly unsized decimal

   The same applies to wires computed from wide registers:
   WRONG:  wire [15:0] baud_next = baud_counter + 16'd1;  // width mismatch
   CORRECT: wire [31:0] baud_next = baud_counter + 1;      // matches reg width
5. No latches: assign defaults to all signals at top of every always @(*) block.
6. Parameters: module {design_name} #(parameter WIDTH=8) (input clk, ...);
7. No expression part-selects: NEVER apply [N:0] slicing to an expression result.
   WRONG: wire [2:0] idx = (bit_counter - 1)[2:0];   // expression part-select — illegal
   WRONG: assign tx = data[(counter - 1)[2:0]];        // same error inside assignment
   CORRECT: wire [3:0] tmp = bit_counter - 1;          // compute in intermediate wire
             wire [2:0] idx = tmp[2:0];                 // THEN slice the wire
   CORRECT: wire [2:0] idx = bit_counter[2:0] - 1;     // slice BEFORE the operation
   The rule: you can slice a named signal, never an expression.
8. ASYNC RESET: active-low means reset when signal is LOW.
   Standard form:  always @(posedge clk or negedge rst_n)
                   if (!rst_n) begin ... end else begin ... end
   AXI variant:    always @(posedge aclk or negedge aresetn)
                   if (!aresetn) begin ... end else begin ... end
   Use whichever signal names match the port list. The STRUCTURE is identical.
   EVERY output reg and internal reg MUST be initialized in the reset block.
   Uninitialized regs produce 'z' (high-impedance) in simulation — a fatal error.
9. SINGLE DRIVER: Every reg driven by exactly ONE always block.
10. COMBINATIONAL FLAGS: If the spec says a flag "goes high WHEN condition",
    implement as: assign flag = (condition); — wire, not reg.
11. ARRAY INDEX WIDTH: When indexing into an N-bit array with a counter,
    the counter MUST be exactly $clog2(N) bits wide — no wider.
    Indexing reg[7:0] requires a 3-bit index (values 0-7).
    Indexing reg[15:0] requires a 4-bit index (values 0-15).
    WRONG: reg [3:0] bit_ctr; assign tx = data[bit_ctr];  // 4-bit index into 8-bit array
    CORRECT option A: reg [2:0] bit_ctr;                  // declare 3-bit counter
    CORRECT option B: assign tx = data[bit_ctr[2:0]];     // slice at point of use
    Verilator ERROR: "Bit extraction of var[7:0] requires 3 bit index, not 4 bits"
    means your counter is too wide for the array it indexes.

[VARIATION SALT: {rtl_salt}]

Return pure JSON:
{{
  "structural_reasoning": "...",
  "verilog_code": "..."
}}
"""
    result = run_llm(prompt, system_prompt, provider, max_api_retries=2, cand_idx=cand_idx)
    # NOTE: RTL_CACHE.put() is intentionally NOT called here.
    # Caching happens in the main loop AFTER all validation checks pass
    # (port match, BLKANDNBLK check, skeleton invariant check).
    # Caching invalid RTL here would cause cache hits to replay failed candidates forever.
    # Attach cache_key so the main loop can write to cache after validation.
    if result and cache_key:
        result["_cache_key"] = cache_key
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
1. First line of the file MUST be: `timescale 1ns/1ps
   (backtick, no space, exactly as shown — this prevents line-1 syntax errors)
2. Verilog-2001 STRICTLY. No SystemVerilog anywhere. Icarus will reject SV syntax.
   WRONG (SystemVerilog — never use):
     wire [7:0] rx_byte = 8'h00;    // initial value on wire — SV only, illegal in V2001
     logic tx_bit;                   // 'logic' type — SV only, use 'reg' or 'wire'
     int loop_var;                   // 'int' type — SV only, use 'integer'
     always_ff @(posedge clk) ...   // SV keyword, use always @(posedge clk)
   CORRECT:
     reg [7:0] rx_byte;             // declare as reg, assign value in initial block
     reg tx_bit;
     integer loop_var;
3. Declare ALL variables at top of module, never inside begin/end blocks.
   Assign initial values in initial begin blocks, NOT on wire/reg declarations.
   WRONG — Icarus rejects these:
     wire [7:0] rx_byte = 8'h00;        // initial value on wire — illegal in Verilog-2001
     initial begin rx_out = 1'b1; end   // driving a wire in initial — use reg instead
     begin reg [7:0] tmp; end           // variable in unnamed begin block — SystemVerilog only
     begin : unnamed begin reg tmp; end // same problem even with colon — must be at module top
     integer i = 0;                     // initializer on integer declaration — illegal
   Icarus error: "Variable declaration in unnamed block requires SystemVerilog"
   Fix: move ALL reg/wire/integer declarations to the TOP of the module, before any always/initial.
   CORRECT:
     reg [7:0] rx_byte;               // plain declaration
     initial begin rx_byte = 8'h00; end // assign in initial block
     integer i;                       // plain declaration
     initial begin i = 0; end         // assign in initial block
4. DUT INSTANTIATION — CRITICAL:
   You MUST instantiate the DUT module with its EXACT name: `{design_name}`
   CORRECT:   {design_name} dut ( .clk(clk), .rst_n(rst_n), ... );
   WRONG:     dut dut ( ... );          // 'dut' is NOT a module name
   WRONG:     my_module dut ( ... );    // use the exact name: {design_name}
   Connect ports by NAME using .portname(signal) — never by position.
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
   NO break OR continue — these are SystemVerilog-only keywords; Icarus rejects them.
   If you need early loop exit, use a named block with disable:
     begin : loop_name
       for (i=0; i<MAX; i=i+1) begin
         if (done) disable loop_name;
         // loop body
       end
     end
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

    # Compute once — used for skeleton injection and invariant checking
    design_type = detect_design_type(clarified_spec, base_prompt)
    if design_type != "generic":
        print(f"   🧩 Design type detected: {design_type} (skeleton available)")

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
    pending_invariant_msgs = []
    empty_rounds           = 0    # consecutive rounds with zero valid RTL candidates
    reviewer_rejected_streak = 0  # consecutive rounds reviewer added/removed blocks without helping

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
            code_hashes           = set()   # fresh arch → fresh candidate space
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
        _port_mismatch_count = 0
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
                    dynamic_idx,
                    design_type,
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
                        _port_mismatch_count += 1
                        continue

                    h = code_hash(v_code)
                    if h in code_hashes:
                        print("   ⚠️  Duplicate candidate — discarded.")
                        continue
                    code_hashes.add(h)

                    # Auto-patch narrow-register == PARAMETER width mismatches
                    v_code = auto_patch_param_width(v_code)

                    # Static BLKANDNBLK check: reject before Verilator wastes a call
                    blk_err = _check_blocking_mix(v_code)
                    if blk_err:
                        print(f"   ⚠️  BLKANDNBLK detected ({blk_err}) — candidate discarded.")
                        continue

                    # Skeleton invariant check: reject if frozen structural lines were removed
                    if design_type != "generic":
                        inv_ok, violations = check_skeleton_invariants(v_code, design_type)
                        if not inv_ok:
                            msg = violations[0][:100]
                            print(f"   ⚠️  Skeleton invariant violated ({msg[:60]}) — discarding.")
                            pending_invariant_msgs.append(msg)
                            continue

                    has_memory_blocks = bool(architecture.get("memory_blocks"))
                    if (architecture.get("module_class") == "DATAPATH"
                            and has_memory_blocks
                            and not re.search(r'reg\s*\[[^\]]+\]\s*\w+\s*\[[^\]]+\]', v_code)):
                        print("   ⚠️  DATAPATH with memory_blocks missing 2D reg. Discarding.")
                        continue

                    # All validations passed — safe to cache now
                    ck = result.get("_cache_key")
                    if ck:
                        RTL_CACHE.put(ck, result)

                    rtl_candidates.append(v_code)

            except concurrent.futures.TimeoutError:
                print("   ⚠️  RTL executor timeout. Restarting round.")

        if rtl_candidates:
            empty_rounds = 0  # reset streak when we get valid candidates
        else:
            empty_rounds += 1

        if not rtl_candidates:
            print("   ❌ No valid RTL candidates this round.")
            # After 3 empty rounds in a row, reset code_hashes and best_v_code
            # to force fresh generation instead of endlessly mutating broken code
            if empty_rounds >= 3:
                print(f"   🔄 {empty_rounds} empty rounds — clearing cache and regenerating fresh.")
                code_hashes.clear()
                # Also evict RTL_CACHE entries for this architecture so cache hits
                # don't re-serve the same broken RTL after the reset.
                # BoundedCache doesn't support selective eviction, so clear entirely.
                RTL_CACHE.cache.clear()
                best_v_code  = None
                empty_rounds = 0
                error_log = (best_err_log or
                             "All mutation attempts produced no valid candidates. Regenerating fresh.")
                continue
            if _port_mismatch_count > 0:
                print("   🔓 Port interface lock cleared (all candidates mismatched).");locked_ports = None;code_hashes.clear()
                error_log = "CRITICAL: All generated RTL had wrong port interface. Match port_interface array exactly — no added, removed, or renamed ports."
            elif pending_invariant_msgs:
                unique_msgs = list(dict.fromkeys(pending_invariant_msgs))
                inv_lines = "\n".join(f"  - {m}" for m in unique_msgs[:3])
                error_log = "STRUCTURAL INVARIANT VIOLATIONS:\n" + inv_lines + "\nFix these first."
                pending_invariant_msgs.clear()
            else:
                code_hashes.clear()
                error_log = "All RTL candidates were empty, had wrong ports, or were duplicates."
            continue

        # ── Reviewer pass ─────────────────────────────────────────────────────
        # Skip reviewer for skeleton-matched designs: the skeleton + invariant
        # checker already enforces structure. Reviewer adds always blocks ~90%
        # of the time on these designs, wasting API calls with no benefit.
        # Skip reviewer for skeleton designs, OR when reviewer has consistently
        # failed to improve candidates (adds/removes blocks every round = wasted API calls)
        skip_reviewer = (design_type != "generic") or (reviewer_rejected_streak >= 3)
        if skip_reviewer:
            reason = ("skeleton design" if design_type != "generic"
                      else f"reviewer unhelpful for {reviewer_rejected_streak} rounds")
            print(f"   🔎 Reviewer: Skipped ({reason}).")
            reviewed_candidates = list(rtl_candidates)
            reviewer_rejected_streak += 1  # keep counting; resets when reviewer helps
        else:
            print(f"   🔎 Reviewer: Scanning {len(rtl_candidates)} RTL candidate(s)...")
            reviewed_candidates = []
        for idx, v_code in enumerate(rtl_candidates if not skip_reviewer else []):
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
                    elif always_after > always_before:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer added always block (likely spurious latch fix). Keeping original.")
                        reviewed_candidates.append(v_code)
                    elif 0.75 < len(fixed) / len(v_code) < 1.3:
                        # Final BLKANDNBLK check on the patched code before accepting
                        blk_err = _check_blocking_mix(fixed)
                        if blk_err:
                            print(f"      ⚠️  Candidate {idx+1}: Reviewer introduced BLKANDNBLK ({blk_err}). Keeping original.")
                            reviewed_candidates.append(v_code)
                        else:
                            print(f"      ✅ Candidate {idx+1}: Patch applied.")
                            reviewed_candidates.append(fixed)
                            reviewer_rejected_streak = 0  # reviewer helped — reset streak
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
                    # Save v_code even at stage 0 so mutation engine has something
                    # to mutate from next round. Remove its hash from code_hashes so
                    # mutations of this base are not blocked as duplicates next round.
                    best_v_code  = v_code
                    code_hashes.discard(code_hash(v_code))
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

        if pending_invariant_msgs:
            unique_inv = list(dict.fromkeys(pending_invariant_msgs))
            inv_note = "\nSTRUCTURAL INVARIANTS VIOLATED (fix alongside sim failure):\n" + "\n".join(f"  - {m}" for m in unique_inv[:3])
            best_err_log = (best_err_log + inv_note) if best_err_log else inv_note.strip()
            pending_invariant_msgs.clear()

        # Clear code_hashes when round produces no simulation pass.
        # This allows mutation to produce variants without being blocked by duplicates.
        # Stage 0 (Verilator fail): fresh generation needed
        # Stage 1 (sim fail): mutation needed — clear so mutated variants aren't
        #   blocked by the hash of the original candidate they're derived from.
        if not round_sim_passed and global_candidate_id > 0:
            code_hashes.clear()

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
