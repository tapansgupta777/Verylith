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
    "gpt-5-mini", "gpt-5",
}

def _is_reasoning_model(model_name: str) -> bool:
    return model_name.strip() in REASONING_MODELS

MAX_TOKENS_FAST   = 4096
MAX_TOKENS_STRONG = 8192

# ==============================================================================
# GLOBAL CLIENT & TRUE LRU CACHES
# ==============================================================================
openai_client = None

class BoundedCache:
    """A True LRU (Least Recently Used) Cache to prevent memory explosion."""
    def __init__(self, max_size=100):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key) # Mark as recently used
            return self.cache[key]
        return None

    def put(self, key, val):
        self.cache[key] = val
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False) # Evict Least Recently Used

    def __contains__(self, key):
        return key in self.cache

ARCH_CACHE = BoundedCache(50)
RTL_CACHE = BoundedCache(200)
TB_CACHE = BoundedCache(200)

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI()
    return openai_client

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def code_hash(text: str) -> str:
    if not text: return ""
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
        if any(kw in line.lower() for kw in ("error", "mismatch", "fail", "syntax", "%warning", "assertion failed")):
            start = max(0, i - 1)
            end   = min(len(lines), i + 3)
            return "\n".join(lines[start:end])
    return truncate_log(log, 5)

def classify_failure(log: str) -> str:
    l = log.lower()
    
    if "timeout" in l or "simulation timeout" in l: return "TB_TIMEOUT"
    if "unoptflat" in l or ("loop" in l and "comb" in l): return "COMBINATIONAL_LOOP"
    if "compile" in l or "syntax" in l: return "SYNTAX_ERROR"
    if "latch" in l: return "LATCH_INFERRED"
    if "width" in l or "size" in l: return "WIDTH_MISMATCH"
        
    if re.search(r'\b[xX]\b', l) or "xxx" in l:
        return "X_PROPAGATION_BUG"
    if "stuck" in l or "never changes" in l:
        return "STATE_MACHINE_STUCK"

    m = re.search(r'expected\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+).*?got\s*[=:]?\s*(?:0x)?([0-9a-fA-F]+)', l)
    if m:
        try:
            base = 16 if ('0x' in l or any(c in 'abcdef' for c in m.group(1).lower() + m.group(2).lower())) else 10
            exp = int(m.group(1), base)
            got = int(m.group(2), base)
            if abs(exp - got) == 1:
                return "OFF_BY_ONE"
        except ValueError:
            pass

    if "priority" in l or "instead of" in l or "order" in l: return "PRIORITY_ORDER_BUG"
    if "shift" in l or "serial" in l or "sipo" in l or "siso" in l or "piso" in l: return "SHIFT_LOGIC"
    if "increment" in l or "count up" in l or "count_up" in l: return "INCREMENT_LOGIC"
    if "decrement" in l or "count down" in l or "count_down" in l: return "DECREMENT_LOGIC"
    if "overflow" in l: return "OVERFLOW_FLAG"
    if "underflow" in l: return "UNDERFLOW_FLAG"
    if "reset" in l: return "RESET_LOGIC"
    if "load" in l: return "LOAD_LOGIC"
    if "assert" in l or "assertion" in l or "invariant" in l: return "PROTOCOL_ASSERTION_FAILED"

    if re.search(r'(expected|mismatch|fail)', l):
        return "LOGIC_MISMATCH"

    return "TB_STRUCTURAL"

def get_error_fingerprint(failure_type: str, full_log: str) -> str:
    primary = extract_primary_error(full_log)
    line_nums = re.findall(r'\.v:(\d+):', primary)
    line_sig = "_".join(line_nums[:3]) if line_nums else code_hash(primary)[:8]
    return f"{failure_type}_{line_sig}"

def localize_bug(v_code: str, failure_type: str) -> str:
    lines = v_code.split('\n')
    snippets = []
    keywords = []
    
    if failure_type == "INCREMENT_LOGIC": keywords = ["+ 1", "+1", "up", "increment", "add"]
    elif failure_type == "DECREMENT_LOGIC": keywords = ["- 1", "-1", "down", "decrement", "sub"]
    elif failure_type == "OVERFLOW_FLAG": keywords = ["overflow", "255", "hff", "'hff", "== 255", "max"]
    elif failure_type == "UNDERFLOW_FLAG": keywords = ["underflow", "== 0", "00"]
    elif failure_type == "RESET_LOGIC": keywords = ["rst", "reset", "0"]
    elif failure_type == "LOAD_LOGIC": keywords = ["load", "data"]
    elif failure_type == "PRIORITY_ORDER_BUG": keywords = ["if", "else if"]
    elif failure_type == "SHIFT_LOGIC": keywords = ["shift", "serial", "{", "}"]
    elif failure_type == "OFF_BY_ONE": keywords = ["<", "<=", ">", ">=", "==", "+", "-"]
    elif failure_type == "PROTOCOL_ASSERTION_FAILED": keywords = ["always", "assign", "ready", "valid", "state"]
    
    if not keywords: return ""

    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in keywords):
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippets.append("\n".join(lines[start:end]).strip())
    
    if snippets:
        unique_snippets = list(dict.fromkeys(snippets)) 
        return "...\n" + "\n...\n".join(unique_snippets[:3]) + "\n..."
    return ""

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
    if not raw_code: return ""
    clean = raw_code.strip()
    triple_tick = "`" * 3
    for fence in (f"{triple_tick}verilog", f"{triple_tick}systemverilog", f"{triple_tick}v", triple_tick):
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
                client = get_openai_client()
                kwargs = {
                    "model":    model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "timeout": 120 if not force_fast_model else 90,
                }
                
                temp_idx = min(cand_idx + attempt, len(temps) - 1)
                
                if is_reasoning:
                    kwargs["max_completion_tokens"] = max_tokens
                    kwargs["reasoning_effort"]      = efforts[temp_idx % 3]
                else:
                    kwargs["max_tokens"]      = max_tokens
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["temperature"]     = temps[temp_idx]

                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling {model_name} (Attempt {attempt+1})...")

                response = client.chat.completions.create(**kwargs)
                content  = response.choices[0].message.content

                if not content or not content.strip():
                    print(f"      ⚠️  Thread {cand_idx+1}: Empty content returned. Retrying...")
                    continue

                return parse_llm_json(content)

        except Exception as e:
            err = str(e).lower()
            if "400" in err and ("unsupported parameter" in err or "invalid_request" in err):
                print(f"   ❌ Thread {cand_idx+1}: Bad request (check model params): {e}")
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
    You are a Lead Hardware Specifications Engineer.
    Translate the user's ambiguous request into a strict, disambiguated technical specification.
    Return pure JSON:
    {
      "formalized_request": "...",
      "data_widths": "...",
      "sequential_or_combinational": "...",
      "clocking_scheme": "...",
      "pipeline_stages": "...",
      "protocol_interfaces": "...",
      "reset_behavior": "...",
      "overflow_behavior": "...",
      "design_constraints": {
         "target_frequency": "...",
         "max_latency": "...",
         "max_area": "..."
      }
    }
    """
    return run_llm(prompt, system_prompt, provider, cand_idx=0, force_fast_model=True, silent=True)

# ==============================================================================
# AGENT: ARCHITECT
# ==============================================================================
def generate_architecture(
    spec:           dict,
    provider:       str,
    previous_failures: str = None,
    attempt_idx:    int = 0,
    forced_style:   str = None,
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
        
    style_directive = f"You MUST strictly adhere to the `{forced_style}` implementation paradigm." if forced_style else ""
    
    salt = random.randint(10000, 99999)
    # PEER UPGRADE: Pipelining via Constraints & Explicit Pointers
    system_prompt = f"""
    You are an elite SoC Micro-Architect. Design a strictly synthesizable hardware architecture.
    {error_context}
    {style_directive}

    CRITICAL: Classify the design correctly:
    - module_class: "FSM" or "DATAPATH"
    - template_type: Select closest standard template ("counter", "fifo", "shift_register", "alu", "fsm", "custom").
    - implementation_style: Define the explicit hardware style used.

    GRAPH DECOMPOSITION REQUIREMENT:
    You must output a structured Datapath Graph instead of text algorithms. 
    1. Read `design_constraints` carefully. If latency > 0 or frequency is high, you MUST define `pipeline_stages` using strictly defined intermediate signals (e.g., `stage1_val = a + b`) to break up combinational paths.
    2. Define `protocol_interfaces` if handling AXI, ready/valid, or Wishbone.
    3. Define `memory_blocks` explicitly. If FIFO/RAM, YOU MUST EXPLICITLY GENERATE `write_ptr`, `read_ptr`, and `count` explicitly in the `pointers` array.
    4. Explicitly define routing (`muxes`) using EXACT `control_conditions` (No natural language strings).

    Return pure JSON exactly matching this schema:
    {{
        "module_class": "DATAPATH",
        "template_type": "fifo",
        "implementation_style": "circular_buffer_with_explicit_pointers",
        "protocol_interfaces": [{{"type": "ready_valid", "signals": ["valid", "ready"]}}],
        "clock_and_reset": {{"clock": "clk", "reset": "rst_n", "active_low": true}},
        "port_interface": [{{"name": "clk", "direction": "input", "width": 1}}],
        "pipeline_stages": [
            {{"stage": 1, "intermediate_signals": ["stage1_sum = a + b"]}}
        ],
        "memory_blocks": [{{"name": "fifo_ram", "type": "dual_port", "width": 8, "depth": 16, "pointers": ["write_ptr", "read_ptr", "count"]}}],
        "registers": [{{"name": "count_reg", "width": 8, "reset_value": "0"}}],
        "datapath_nodes": [
            {{"op": "+", "dst": "next_count", "a": "count_reg", "b": "1"}}
        ],
        "muxes": [
            {{
                "dst": "count_next_final", 
                "control_conditions": [
                    {{"when": "load_en", "route": "load_data"}},
                    {{"default": "count_reg"}}
                ]
            }}
        ],
        "flags": [
            {{"name": "overflow", "type": "combinational", "logic": "count_reg == 255 && enable && up_down"}}
        ],
        "fsm_specific": {{"state_encoding": {{}}, "next_state_logic": []}}
    }}

    [VARIATION SALT: {salt}]
    """
    result = run_llm(json.dumps(spec), system_prompt, provider, cand_idx=attempt_idx, force_fast_model=True, silent=True)
    if result:
        ARCH_CACHE.put(cache_key, result)
    return result

def validate_architecture(arch: dict, expected_min_ports: int = 3) -> bool:
    if not arch or "module_class" not in arch: return False
    if not arch.get("clock_and_reset"): return False
    ports = arch.get("port_interface", [])
    if not ports or len(ports) < expected_min_ports: return False
    if not arch.get("registers") and not arch.get("datapath_nodes") and not arch.get("combinational_logic"): return False
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
      - If the code is clean, return {"status": "PASSED", "fixed_code": ""}.
      - If patching, return {"status": "REJECTED", "fixed_code": "<full corrected module>"}.
    Return pure JSON: {"status": "...", "fixed_code": "..."}
    """
    return run_llm(v_code, system_prompt, provider, cand_idx=0, force_fast_model=True, silent=True)

# ==============================================================================
# AGENT: RTL GENERATOR
# ==============================================================================
def generate_rtl(
    prompt:         str,
    provider:       str,
    design_name:    str,
    architecture:   dict,
    previous_ports: str = None,
    previous_error: str = None,
    best_v_code:    str = None,
    cand_idx:       int = 0,
) -> dict:
    
    cache_key = get_cache_key(prompt, architecture, previous_error, best_v_code, cand_idx)
    if cache_key in RTL_CACHE:
        print(f"      ⚡ Cache Hit: Reusing generated RTL for Thread {cand_idx+1}.")
        return RTL_CACHE.get(cache_key)

    port_rule = f"2. STRICT INTERFACE LOCK: You MUST reuse this exact port list verbatim:\n   {previous_ports}" if previous_ports else "2. STRICT INTERFACE: Port declarations MUST perfectly match the Architect's `port_interface` array."

    error_context = ""
    if previous_error and best_v_code:
        failure_type  = classify_failure(previous_error)
        localization = localize_bug(best_v_code, failure_type)
        loc_str = f"\nBUG LOCALIZED TO THIS CODE (Modify ONLY this logic):\n{localization}\n" if localization else ""
        
        error_context = f"""
        MUTATION REQUIRED — previous RTL failed.
        FAILURE CLASSIFICATION: {failure_type}
        SIMULATION OUTPUT:
        {previous_error}
        {loc_str}

        TARGETED FIX RULES based on failure type:
        - COMBINATIONAL_LOOP → Break the continuous cyclic dependency.
        - OFF_BY_ONE       → Check boundary conditions (< vs <=) or counter initialization.
        - PRIORITY_ORDER_BUG → Reorder `if/else` statements.
        - PROTOCOL_ASSERTION_FAILED → The testbench caught an invariant violation. Fix the handshake timing (e.g. valid/ready).
        - LOGIC_MISMATCH   → Trace the failing signal back to its driver.
        - X_PROPAGATION_BUG → Trace the signal and explicitly initialize it inside the reset block.

        You MUST ONLY modify logic related to the failure type above. Do not touch unrelated blocks.
        PREVIOUS BEST VERILOG (mutate this, do not rewrite from scratch):
        {best_v_code}
        """
    elif previous_error:
        error_context = f"SYNTAX ERROR IN PREVIOUS ATTEMPT:\n{previous_error}\nFix only the line causing this error. Do not restructure the module."

    alg_sections = []
    
    if architecture.get("protocol_interfaces"): alg_sections.append("PROTOCOL ABSTRACTIONS:\n  " + str(architecture["protocol_interfaces"]))
    if architecture.get("pipeline_stages"): alg_sections.append("PIPELINE STAGES (Ensure correct clock latency by explicitly creating intermediate registers for these boundary signals):\n  " + str(architecture["pipeline_stages"]))
    if architecture.get("memory_blocks"): alg_sections.append("MEMORY ARCHITECTURE (Explicitly implement RAM/FIFO logic using the provided pointers):\n  " + str(architecture["memory_blocks"]))

    if architecture.get("datapath_nodes"): 
        lines = [f"  Node '{n.get('dst')}': {n.get('a')} {n.get('op')} {n.get('b')}" for n in architecture["datapath_nodes"]]
        alg_sections.append("DATAPATH NODES (Implement as continuous assigns or combinational blocks):\n  " + "\n  ".join(lines))
    
    if architecture.get("muxes"): 
        lines = []
        for m in architecture["muxes"]:
            dst = m.get("dst", "unknown_dst")
            lines.append(f"  MUX Target '{dst}':")
            for cond in m.get("control_conditions", []):
                if "default" in cond:
                    lines.append(f"    else -> {cond['default']}")
                else:
                    lines.append(f"    if ({cond.get('when')}) -> {cond.get('route')}")
        alg_sections.append("MULTIPLEXERS & ROUTING (Implement strictly as prioritized if/else blocks):\n" + "\n".join(lines))
        
    if architecture.get("flags"): 
        lines = [f"  FLAG '{f.get('name')}' ({f.get('type')}): {f.get('logic')}" for f in architecture["flags"]]
        alg_sections.append("FLAGS:\n  " + "\n  ".join(lines))
    
    if architecture.get("module_class") == "FSM":
        alg_sections.append("FSM STRUCTURE: You MUST use a `case(state)` statement for state transitions, and cleanly separate `state_reg` and `next_state`.")
    
    algorithm_section = f"\nSYNTAX COMPILER DIRECTIVE — You are a strict syntax compiler. Map the datapath nodes and muxes EXACTLY as structured below into Verilog. Do not invent new logic structures or bypass the pipeline stages:\n" + "\n\n".join(alg_sections)

    system_prompt = f"""
    You are an elite RTL design engineer acting as a strict syntax compiler. Generate ONLY the synthesizable Verilog RTL module.
    Do NOT generate a testbench — a separate agent handles that.
    {error_context}{algorithm_section}

    SILICON RULES — all mandatory:
    1. Top module named exactly: `{design_name}`
    {port_rule}
    3. Verilog-2001 only. No SystemVerilog.
    4. Sequential always @(posedge clk): non-blocking (<=) only. Combinational always @(*): blocking (=) only.
    5. No latches: assign defaults to all signals at top of every always @(*) block.
    6. ASYNC RESET: active-low means reset when signal is LOW. Sensitivity: always @(posedge clk or negedge rst_n).
    7. SINGLE DRIVER: Every reg driven by exactly ONE always block.

    Return pure JSON:
    {{ "structural_reasoning": "...", "verilog_code": "..." }}
    """
    result = run_llm(prompt, system_prompt, provider, max_api_retries=2, cand_idx=cand_idx)
    if result and result.get("verilog_code"):
        RTL_CACHE.put(cache_key, result)
    return result

# ==============================================================================
# AGENT: TESTBENCH GENERATOR
# ==============================================================================
def generate_testbench(
    prompt:         str,
    provider:       str,
    design_name:    str,
    port_interface: str, 
    architecture:   dict, # PEER UPGRADE: Passing architecture for Protocol Assertions
    previous_error: str = None,
    previous_tb:    str = None,
    cand_idx:       int = 0,
) -> str:
    
    cache_key = get_cache_key(prompt, port_interface, previous_error, previous_tb, cand_idx)
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

    # PEER UPGRADE: Automated Protocol Assertion Injection
    protocol_assertions = ""
    if architecture and architecture.get("protocol_interfaces"):
        protocols = str(architecture.get("protocol_interfaces")).lower()
        if "ready" in protocols or "valid" in protocols:
            protocol_assertions = """
    7. PROTOCOL INVARIANTS: The Architecture specifies a handshake protocol. You MUST include this concurrent assertion:
       `always @(posedge clk) if (valid && !ready && $past(valid)) if (data !== $past(data)) $display("ASSERTION FAILED: Data mutated while valid without ready");`
       *(Note: Construct a 1-cycle delay register for history if $past is unsupported).*
            """

    system_prompt = f"""
    You are a hardware verification engineer. Write a Verilog-2001 testbench for a module.
    YOU DO NOT HAVE ACCESS TO THE RTL CODE. You must write a Black-Box Testbench based purely on the Spec and Port Interface.

    ORIGINAL SPEC:
    {prompt}

    PORT INTERFACE (Instantiate the DUT precisely using these signals):
    {port_interface}

    TESTBENCH RULES — all mandatory:
    1. `timescale 1ns/1ps`
    2. Verilog-2001 only. No SystemVerilog anywhere.
    3. Declare ALL variables at top of module.
    4. Clock: always #5 clk = ~clk; with initial clk = 0;
    5. TIMING (CRITICAL): Apply stimulus at negedge clk, sample at the NEXT negedge clk.
       Once you sample, apply the next stimulus IMMEDIATELY. Do NOT add extra @(negedge clk) delays between checks and new stimulus!
    6. COVERAGE-DRIVEN RANDOM STIMULUS: Use loops (e.g., `repeat(100)`). MUST declare `integer seed = 12345;` and use `$random(seed)`.
       Track expected state internally in a Golden Model and assert `!==` against DUT outputs cycle-by-cycle.
       **COVERAGE BINS**: You MUST use `$display("COVER_HIT: <scenario>");` when edge cases occur (e.g., wrap-around, full, empty). Python will parse these to measure coverage.
    {protocol_assertions}
    8. NO WHILE LOOPS. Use only bounded repeat() or for loops.
    9. EVERY code path must call $finish. 
    10. Success: $display("SIMULATION_SUCCESS"); $finish;
    11. Timeout: initial begin #100000; $display("FAIL: Timeout"); $finish; end

    {error_context}

    Return pure JSON:
    {{ "testbench_code": "..." }}
    """
    result = run_llm(prompt, system_prompt, provider, max_api_retries=2, cand_idx=cand_idx)
    tb_code = clean_code_string(result.get("testbench_code", ""))
    if tb_code:
        TB_CACHE.put(cache_key, tb_code)
    return tb_code

# ==============================================================================
# PORT MATCHING
# ==============================================================================
def ports_match(v_code: str, locked_ports_json: str) -> bool:
    if not locked_ports_json: return True
    locked_ports = json.loads(locked_ports_json)
    
    m = re.search(r'module\s+\w+\s*(?:#\s*\(.*?\))?\s*\((.*?)\);', v_code, re.S)
    if not m: return False
    
    port_str = re.sub(r'\s+', ' ', m.group(1))
    matches = re.findall(r'(input|output|inout)\s+(?:wire\s+|reg\s+)?(?:signed\s+)?(\[[^\]]+\])?\s*(\w+)', port_str)
    rtl_ports = {m[2]: {"dir": m[0], "width": m[1].strip() if m[1] else ""} for m in matches}

    for p in locked_ports:
        name = str(p.get("name", ""))
        direction = str(p.get("direction", ""))
        arch_width = str(p.get("width", "1"))

        if name not in rtl_ports: 
            return False
        if rtl_ports[name]["dir"] != direction: 
            return False
            
        rtl_w = rtl_ports[name]["width"].replace(" ", "")
        
        expected_w = ""
        if arch_width.isdigit():
            w_int = int(arch_width)
            if w_int > 1:
                expected_w = f"[{w_int-1}:0]"
        else:
            expected_w = arch_width.replace(" ", "")
            
        if rtl_w != expected_w:
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
    cmd = ["verilator", "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Wno-UNUSED", "-Wno-UNDRIVEN", "-Werror-WIDTH", "-Werror-IMPLICIT", v_file]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if "UNOPTFLAT" in r.stderr: return False, "COMBINATIONAL LOOP DETECTED (UNOPTFLAT)."
        if r.returncode == 0: return True, "Clean"
        return False, truncate_log(r.stderr)
    except subprocess.TimeoutExpired:
        return False, "Verilator timeout."

def verify_with_iverilog(v_file: str, tb_file: str, design_name: str, candidate_id: int):
    workspace  = f"./workspace/{design_name}/candidate_{candidate_id}"
    output_bin = f"{workspace}/{design_name}_sim.vvp"

    cr = subprocess.run(["iverilog", "-o", output_bin, v_file, tb_file], capture_output=True, text=True)
    if cr.returncode != 0: return False, f"Compile failed:\n{truncate_log(cr.stderr)}", 0

    try:
        sr = subprocess.run(["vvp", output_bin], capture_output=True, text=True, timeout=60)
        
        # PEER UPGRADE: Python-based Coverage Parsing via Testbench output
        cover_hits = len(set(re.findall(r'COVER_HIT:\s*(\w+)', sr.stdout)))
        
        if sr.returncode != 0: return False, f"Runtime failed:\n{truncate_log(sr.stderr)}", cover_hits
        if any(kw in sr.stdout for kw in ("ERROR", "Error", "FAIL", "Fail", "ASSERTION FAILED")): return False, f"Verification failed:\n{truncate_log(sr.stdout, 15)}", cover_hits
        if "SIMULATION_SUCCESS" not in sr.stdout: return False, "Missing SIMULATION_SUCCESS — simulation exited prematurely.", cover_hits
        return True, "Simulation clean.", cover_hits
    except subprocess.TimeoutExpired:
        return False, "Simulation timeout (possible infinite loop).", 0

def verify_with_yosys(v_file: str, base_prompt: str):
    cmd = ["yosys", "-p", f"read_verilog {v_file}; proc; opt; fsm; memory; check -assert; prep; stat"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        log = r.stderr + "\n" + r.stdout
        if r.returncode == 0 and "ERROR" not in log:
            dffs  = len(re.findall(r'\$(?:a?dff|dffe|sdff)', log, re.IGNORECASE))
            cells = re.search(r'Number of cells:\s+(\d+)', log)
            
            seq_keywords = ["clk", "clock", "sync", "sequential", "posedge", "negedge"]
            if dffs == 0 and any(kw in base_prompt.lower() for kw in seq_keywords):
                return False, "CRITICAL: Sequential logic expected but 0 registers synthesized (Latches/Logic optimized away)."
                
            return True, f"Cells: {cells.group(1) if cells else '?'}, DFFs: {dffs}"
        return False, truncate_log(log, 20)
    except subprocess.TimeoutExpired:
        return False, "Yosys timeout."

def verify_gate_level(v_file: str, tb_file: str, design_name: str, candidate_id: int):
    workspace = f"./workspace/{design_name}/candidate_{candidate_id}"
    gate_v = f"{workspace}/{design_name}_gate.v"
    output_bin = f"{workspace}/{design_name}_gate_sim.vvp"

    synth_cmd = ["yosys", "-p", f"read_verilog {v_file}; synth -top {design_name} -flatten; proc; opt; dfflibmap -liberty +/cmos/cells.lib; abc -liberty +/cmos/cells.lib; opt; write_verilog -noattr {gate_v}"]
    synth_r = subprocess.run(synth_cmd, capture_output=True, text=True)
    if synth_r.returncode != 0: return False, f"Gate-level synthesis failed:\n{truncate_log(synth_r.stderr)}"

    comp_cmd = ["iverilog", "-o", output_bin, gate_v, tb_file]
    comp_r = subprocess.run(comp_cmd, capture_output=True, text=True)
    if comp_r.returncode != 0: return False, f"Gate-level compile failed:\n{truncate_log(comp_r.stderr)}"

    try:
        sim_r = subprocess.run(["vvp", output_bin], capture_output=True, text=True, timeout=60)
        if sim_r.returncode != 0 or "FAIL" in sim_r.stdout or "ERROR" in sim_r.stdout:
            return False, f"Gate-level verification failed:\n{truncate_log(sim_r.stdout)}"
        if "SIMULATION_SUCCESS" not in sim_r.stdout: return False, "Gate-level simulation exited prematurely."
        return True, "Gate-Level Simulation Clean"
    except subprocess.TimeoutExpired:
        return False, "Gate-level simulation timeout."

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
    if not clarified_spec: return False

    arch_hashes           = set()
    code_hashes           = set()
    locked_ports          = None
    
    arch_failures         = {}
    error_log             = None
    architecture          = None
    architecture_flawed   = True
    invalid_arch_count    = 0
    duplicate_arch_count  = 0

    best_score            = (-1, -999999)
    best_v_code           = None
    best_tb_code          = None
    best_err_log          = None
    global_candidate_id   = 0
    consecutive_sim_fails = 0
    last_err_fingerprint  = None   
    same_err_count        = 0

    # PEER UPGRADE: Deterministic Architecture Diversity Engine
    ARCH_STYLES = [
        "behavioral_single_block",
        "explicit_datapath_and_controller",
        "pipelined_stages",
        "decoupled_control",
        "register_chain_optimized"
    ]

    for attempt in range(max_retries):
        print(f"\n{'─'*60}")
        print(f"🔄 Generation {attempt + 1}/{max_retries}")
        print(f"{'─'*60}")

        trigger_redesign = False
        if consecutive_sim_fails >= 3:
            trigger_redesign = True
            print("   🔄 3 consecutive sim failures — forcing full architecture redesign...")
        elif architecture:
            current_arch_hash = code_hash(json.dumps(architecture, sort_keys=True))
            if len(arch_failures.get(current_arch_hash, set())) >= 3:
                trigger_redesign = True
                print("   🔄 3 distinct logic failure types detected in this architecture — forcing full architecture redesign...")

        if trigger_redesign:
            architecture_flawed   = True
            consecutive_sim_fails = 0
            best_score            = (-1, -999999)
            best_v_code           = None
            best_tb_code          = None
            locked_ports          = None
            
            fail_history_str = "\n".join([f"- Arch Hash {h[:6]}: " + ", ".join(reasons) for h, reasons in arch_failures.items()])
            error_log = f"CRITICAL: Previous architectures failed.\nFailure History:\n{fail_history_str}\nProduce a structurally DIFFERENT blueprint."

        if architecture_flawed:
            # Cycle through diverse architectural styles
            enforced_style = ARCH_STYLES[attempt % len(ARCH_STYLES)]
            architecture = generate_architecture(clarified_spec, provider, error_log, attempt, enforced_style)
            arch_hash    = code_hash(json.dumps(architecture, sort_keys=True))

            if not validate_architecture(architecture):
                invalid_arch_count += 1
                if invalid_arch_count > 5:
                    print("   ❌ FATAL: Architect agent is stuck in an invalid schema loop. Aborting.")
                    return False
                print(f"   ⚠️  Architect returned invalid blueprint. Retrying...")
                error_log = "CRITICAL: Blueprint is missing required fields. Ensure datapath_nodes and muxes are defined. Retry with complete schema."
                continue

            if arch_hash in arch_hashes:
                duplicate_arch_count += 1
                if duplicate_arch_count > 5:
                    print("   ⚠️  Architect stuck in duplicate loop. Forcing acceptance to break local minimum.")
                    architecture_flawed = False
                    duplicate_arch_count = 0
                else:
                    print("   ⚠️  Architect produced a duplicate blueprint. Forcing variation...")
                    error_log = "CRITICAL: Do not repeat previous blueprints. Change the architecture."
                    continue
            else:
                arch_hashes.add(arch_hash)
                architecture_flawed = False
                duplicate_arch_count = 0
                invalid_arch_count = 0
                print(f"   📐 Architect: New blueprint accepted (class={architecture.get('module_class', '?')}, style={enforced_style})")
        else:
            print("   📐 Architect: Reusing stable blueprint.")

        # ── RTL generation (parallel) ─────────────────────────────────────────
        if best_v_code: print(f"   🧬 Evolution Engine: Mutating best RTL (score={best_score[0]}/4)...")
        else: print(f"   🛠️  Generator: Spawning {candidates_per_round} parallel RTL candidates...")

        rtl_candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidates_per_round) as executor:
            futures = {}
            for i in range(candidates_per_round):
                dynamic_cand_idx = i + same_err_count
                futures[executor.submit(
                    generate_rtl, base_prompt, provider, design_name, architecture, locked_ports, best_err_log or error_log, best_v_code, dynamic_cand_idx
                )] = i

            try:
                for future in concurrent.futures.as_completed(futures, timeout=240):
                    try:
                        result = future.result(timeout=150)
                    except Exception:
                        continue

                    v_code = clean_code_string(result.get("verilog_code", ""))
                    if len(v_code) < 50: continue
                    if locked_ports and not ports_match(v_code, locked_ports): continue
                    
                    h = code_hash(v_code)
                    if h in code_hashes: continue
                    code_hashes.add(h)

                    has_memory_blocks = bool(architecture.get("memory_blocks"))
                    if (architecture.get("module_class") == "DATAPATH"
                            and has_memory_blocks
                            and not re.search(r'reg\s*\[[^\]]+\]\s*\w+\s*\[[^\]]+\]', v_code)):
                        print("   ⚠️  DATAPATH with memory_blocks missing 2D reg array. Discarding.")
                        continue

                    rtl_candidates.append(v_code)

            except concurrent.futures.TimeoutError:
                print("   ⚠️  Executor timeout — API likely hung. Restarting round.")

        if not rtl_candidates:
            print("   ❌ No valid RTL candidates this round.")
            error_log = "All RTL candidates were empty, had wrong ports, or were duplicates."
            continue

        # ── Reviewer pass ─────────────────────────────────────────────────────
        print(f"   🔎 Reviewer: Scanning {len(rtl_candidates)} RTL candidate(s) for latches/loops...")
        reviewed_candidates = []
        for idx, v_code in enumerate(rtl_candidates):
            review = review_hardware(v_code, provider)
            if review.get("status") == "REJECTED":
                fixed = clean_code_string(review.get("fixed_code", ""))
                
                if fixed and "module " in fixed and "endmodule" in fixed and design_name in fixed:
                    has_always_before = len(re.findall(r'\balways\b', v_code))
                    has_always_after  = len(re.findall(r'\balways\b', fixed))
                    
                    if has_always_after < has_always_before:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer removed `always` block. Rejecting patch.")
                        reviewed_candidates.append(v_code)
                    elif 0.75 < len(fixed) / len(v_code) < 1.3:
                        print(f"      ✅ Candidate {idx+1}: Patch applied safely.")
                        reviewed_candidates.append(fixed)
                    else:
                        print(f"      ⚠️  Candidate {idx+1}: Hallucinated rewrite. Keeping original.")
                        reviewed_candidates.append(v_code)
                else:
                    print(f"      ⚠️  Candidate {idx+1}: Reviewer broke module structure. Rejecting patch.")
                    reviewed_candidates.append(v_code)
            else:
                reviewed_candidates.append(v_code)

        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture["port_interface"])
            print(f"   🔒 Port interface locked ({len(architecture['port_interface'])} ports).")

        # ── Testbench generation ──────────────────────────────────────────────
        _structural_failures = {"TB_TIMEOUT", "SYNTAX_ERROR", "TB_STRUCTURAL", "COMBINATIONAL_LOOP"}
        tb_failure_type = classify_failure(best_err_log) if best_err_log else None
        tb_is_logic_failure = (best_tb_code is not None and tb_failure_type is not None and tb_failure_type not in _structural_failures)

        print(f"   📋 Testbench Agent: ", end="")
        if tb_is_logic_failure: print(f"Reusing locked testbench.")
        else: print(f"Generating {len(reviewed_candidates)} testbench(es) in parallel...")

        def _gen_tb(args):
            idx, v_code = args
            if tb_is_logic_failure: return idx, v_code, best_tb_code
            
            port_interface = locked_ports if locked_ports else json.dumps(architecture.get("port_interface"))
            tb = generate_testbench(base_prompt, provider, design_name, port_interface, architecture, best_err_log if best_tb_code else None, best_tb_code, idx)
            return idx, v_code, tb

        candidates = []  
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(reviewed_candidates)) as tb_executor:
            tb_futures = [tb_executor.submit(_gen_tb, (idx, v_code)) for idx, v_code in enumerate(reviewed_candidates)]
            for tb_future in concurrent.futures.as_completed(tb_futures, timeout=180):
                try:
                    idx, v_code, tb_code = tb_future.result(timeout=120)
                    if tb_code and len(tb_code) > 50:
                        status = "🔒 Locked" if tb_is_logic_failure else "✅ Generated"
                        print(f"      {status} Testbench {idx+1}.")
                        candidates.append((v_code, tb_code))
                except Exception: continue

        if not candidates:
            print("   ❌ No valid candidates after testbench generation.")
            continue

        # ── Verification ──────────────────────────────────────────────────────
        round_passed     = False
        round_sim_passed = False

        for v_code, tb_code in candidates:
            global_candidate_id += 1
            print(f"\n   🔬 Testing Candidate #{global_candidate_id}...")

            v_file, tb_file = save_to_workspace(design_name, v_code, tb_code, global_candidate_id)
            current_score = (0, 0)

            passed, log = verify_with_verilator(v_file, design_name, v_code)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Verilator: {err}")
                current_score = (0, -len(err))
                if current_score > best_score:
                    best_score   = current_score
                    best_err_log = f"Syntax Error:\n{err}"
            else:
                print("      ✅ Verilator passed.")
                passed, log, cover_hits = verify_with_iverilog(v_file, tb_file, design_name, global_candidate_id)
                if not passed:
                    err              = extract_primary_error(log)
                    full_sim_context = truncate_log(log, 30)
                    failure_type     = classify_failure(full_sim_context)
                    print(f"      ❌ Icarus [{failure_type}]: {err}")
                    
                    err_fp = get_error_fingerprint(failure_type, full_sim_context)
                    if err_fp == last_err_fingerprint: same_err_count += 1
                    else:
                        last_err_fingerprint = err_fp
                        same_err_count       = 1

                    if same_err_count >= 3:
                        print(f"      🔁 Same error repeated {same_err_count}x — Boosting Temp & Unlocking TB.")
                        best_tb_code   = None   
                        same_err_count = 0

                    mismatches = len(re.findall(r'(fail|error|mismatch|expected|assert)', log, re.IGNORECASE))
                    # PEER UPGRADE: Coverage-Rewarded Partial Fitness Score
                    current_score = (1, (cover_hits * 10) - mismatches)
                    
                    if current_score > best_score:
                        best_score   = current_score
                        best_v_code  = v_code
                        best_tb_code = tb_code 
                        best_err_log = f"FAILURE TYPE: {failure_type}\nSimulation Error (full output):\n{full_sim_context}"
                    
                    if architecture:
                        current_arch_hash = code_hash(json.dumps(architecture, sort_keys=True))
                        if current_arch_hash not in arch_failures:
                            arch_failures[current_arch_hash] = set()
                        arch_failures[current_arch_hash].add(failure_type)
                else:
                    round_sim_passed = True
                    print(f"      ✅ Icarus simulation passed. (Coverage Hits: {cover_hits})")
                    passed, log = verify_with_yosys(v_file, base_prompt)
                    if not passed:
                        err = extract_primary_error(log)
                        print(f"      ❌ Yosys Synthesis: {err}")
                        current_score = (2, 0)
                        if current_score > best_score:
                            best_score   = current_score
                            best_v_code  = v_code
                            best_tb_code = tb_code
                            best_err_log = f"Synthesis Error:\n{err}"
                    else:
                        print(f"      ✅ Yosys Synthesis passed: {log}")
                        print("      ⚙️  Running Gate-Level Simulation...")
                        passed, log = verify_gate_level(v_file, tb_file, design_name, global_candidate_id)
                        if not passed:
                            print(f"      ❌ Gate-Level Sim: {log}")
                            current_score = (3, 0)
                            if current_score > best_score:
                                best_score   = current_score
                                best_v_code  = v_code
                                best_tb_code = tb_code
                                best_err_log = f"Gate-Level Simulation Error:\n{log}"
                        else:
                            current_score = (4, 0)
                            print(f"      ✅ Gate-Level Simulation passed.")
                            round_passed = True
                            break
            
            if current_score < best_score and attempt > 0:
                cand_dir = os.path.dirname(os.path.dirname(v_file))
                if os.path.exists(cand_dir) and "candidate_" in cand_dir:
                    shutil.rmtree(cand_dir, ignore_errors=True)

        if round_passed:
            print(f"\n{'═'*60}")
            print("🎉 PIPELINE COMPLETE — Structural + Functional + Gate-Level Verification passed.")
            print(f"📁 Winning design: ./workspace/{design_name}/candidate_{global_candidate_id}/")
            print(f"{'═'*60}\n")
            return True

        if best_score[0] <= 1 or not round_sim_passed:
            consecutive_sim_fails += 1
        else:
            consecutive_sim_fails = 0

        print(f"\n   📊 Round summary: best_score=Stage {best_score[0]} | "
              f"sim_fails_streak={consecutive_sim_fails} | "
              f"candidates_tested={global_candidate_id}")

    print("\n🚨 Max retries reached without a passing design.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChipGPT: Autonomous RTL Generator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("prompt", type=str,
        help="Natural language description of the hardware to generate.")
    parser.add_argument("--name", type=str, default="my_module",
        help="Exact name of the top-level Verilog module (default: my_module).")
    parser.add_argument("--provider", type=str, choices=["openai", "groq"], default="openai",
        help="LLM provider to use (default: openai).")
    parser.add_argument("--retries", type=int, default=10,
        help="Maximum number of generation attempts (default: 10).")
    parser.add_argument("--candidates", type=int, default=3,
        help="Parallel RTL candidates per round (default: 3).")
    args = parser.parse_args()

    success = autonomous_build_loop(
        base_prompt=args.prompt,
        design_name=args.name,
        provider=args.provider,
        max_retries=args.retries,
        candidates_per_round=args.candidates,
    )
    exit(0 if success else 1)
