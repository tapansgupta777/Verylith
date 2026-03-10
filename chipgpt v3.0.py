import os
import json
import argparse
import subprocess
import hashlib
import re
import time
import random
import concurrent.futures
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
MAX_TOKENS_STRONG = 16384

# ==============================================================================
# GLOBAL CLIENT
# ==============================================================================
openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI()
    return openai_client

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================
def code_hash(code: str) -> str:
    normalized = re.sub(r'\s+', ' ', code.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def truncate_log(log: str, max_lines: int = 15) -> str:
    lines = log.strip().split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + "\n... [ERRORS TRUNCATED]"
    return log

def extract_primary_error(log: str) -> str:
    lines = log.strip().split('\n')
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ("error", "mismatch", "fail", "syntax")):
            start = max(0, i - 1)
            end   = min(len(lines), i + 2)
            return "\n".join(lines[start:end])
    return truncate_log(log, 5)

def classify_failure(log: str) -> str:
    """Classify simulation failure into a broad category for targeted mutation."""
    l = log.lower()
    # Timeout/infinite loop — testbench structural problem, not RTL logic
    if "timeout" in l or "simulation timeout" in l:
        return "TB_TIMEOUT"
    if "compile" in l or "syntax" in l:
        return "SYNTAX_ERROR"
    # Off-by-one: expected and got values differ by exactly 1
    # Match patterns like "expected 0xa7 got 0xa8" or "expected=3 got=4"
    m = re.search(r'expected\s*[=:]?\s*(?:0x)?([0-9a-f]+).*?got\s*[=:]?\s*(?:0x)?([0-9a-f]+)', l)
    if m:
        try:
            exp = int(m.group(1), 16)
            got = int(m.group(2), 16)
            if abs(exp - got) == 1:
                return "OFF_BY_ONE"
        except ValueError:
            pass
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
    return "UNKNOWN"

def parse_llm_json(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        print("   ⚠️  LLM returned empty response (possible truncation or API error).")
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
    temps   = [0.2, 0.5, 0.8]
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
                    "timeout": 180 if not force_fast_model else 90,
                }
                if is_reasoning:
                    kwargs["max_completion_tokens"] = max_tokens
                    kwargs["reasoning_effort"]      = efforts[(cand_idx + attempt) % 3]
                else:
                    kwargs["max_tokens"]      = max_tokens
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["temperature"]     = temps[(cand_idx + attempt) % 3]

                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling {model_name} (Attempt {attempt+1})...")

                response = client.chat.completions.create(**kwargs)
                content  = response.choices[0].message.content

                if not content or not content.strip():
                    print(f"      ⚠️  Thread {cand_idx+1}: Empty content returned. Retrying...")
                    continue

                return parse_llm_json(content)

            elif provider == "groq":
                client = Groq()
                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling Groq llama-3.3-70b...")
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=temps[(cand_idx + attempt) % 3],
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
  "latency_expectation": "...",
  "reset_behavior": "...",
  "overflow_behavior": "..."
}
"""
    return run_llm(prompt, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)

# ==============================================================================
# AGENT: ARCHITECT
# ==============================================================================
def generate_architecture(
    spec:           dict,
    provider:       str,
    previous_error: str = None,
    attempt_idx:    int = 0,
) -> dict:
    print(f"   📐 Routing to ARCHITECT AGENT ({provider.upper()}) [{FAST_MODEL}]...")
    error_context = (
        f"\nCRITICAL FEEDBACK FROM PREVIOUS RUN:\n{previous_error}\n"
        if previous_error else ""
    )
    salt = random.randint(10000, 99999)
    system_prompt = f"""
You are an elite SoC Architect. Design a strictly synthesizable hardware architecture.
{error_context}

CRITICAL: Classify the design correctly:
- DATAPATH: counters, FIFOs, ALUs, datapaths, memories, shift registers, arithmetic units.
- FSM: protocol controllers, sequence detectors, handshake controllers, state machines.
A counter with overflow/underflow flags is always DATAPATH, never FSM.

You MUST list ALL ports from the specification in port_interface.
A module with N signals in the spec needs N entries in port_interface.

ALGORITHM DESCRIPTION REQUIREMENT:
You must fully specify the behavioral algorithm, not just list registers.
The algorithm description must be detailed enough that an RTL engineer can write
Verilog by transcription, with NO invention required.

For every register in `registers`, describe its exact next-state logic in `control_logic`.
For every output flag, describe its exact combinational condition in `flag_logic`.
Use pseudocode like:
  "if load_en → next_count = load_data
   else if enable and up_down → next_count = count + 1
   else if enable and !up_down → next_count = count - 1
   else → next_count = count"

For `priority_order`, list control signals in strict evaluation order (highest priority first).

Return pure JSON exactly matching this schema:
{{
    "module_class": "FSM" or "DATAPATH",
    "clock_and_reset": {{"clock": "clk", "reset": "rst_n", "active_low": true}},
    "port_interface": [{{"name": "clk", "direction": "input", "width": 1}}],
    "registers": [{{"name": "...", "width": 8, "reset_value": "0"}}],
    "memory_blocks": [{{"name": "...", "width": 8, "depth": 16}}],
    "datapath_signals": [{{"name": "...", "width": 8}}],
    "control_logic": [
        {{
            "register": "count",
            "algorithm": "if load_en → count = load_data; else if enable and up_down → count = count + 1; else if enable and !up_down → count = count - 1; else count unchanged"
        }}
    ],
    "flag_logic": [
        {{
            "signal": "overflow",
            "type": "combinational",
            "condition": "count == 255 AND enable AND up_down"
        }},
        {{
            "signal": "underflow",
            "type": "combinational",
            "condition": "count == 0 AND enable AND !up_down"
        }}
    ],
    "priority_order": ["rst_n (async, active-low)", "load_en", "enable"],
    "fsm_specific": {{"state_encoding": {{}}, "state_transitions": []}}
}}

[VARIATION SALT: {salt}]
"""
    return run_llm(json.dumps(spec), system_prompt, provider,
                   cand_idx=attempt_idx, force_fast_model=True, silent=True)

# ==============================================================================
# VALIDATE ARCHITECTURE
# ==============================================================================
def validate_architecture(arch: dict, expected_min_ports: int = 3) -> bool:
    if not arch or "module_class" not in arch:
        return False
    ports = arch.get("port_interface", [])
    if not ports or len(ports) < expected_min_ports:
        return False
    port_names = [p.get("name", "").lower() for p in ports]
    if not any("clk" in n or "clock" in n for n in port_names):
        return False
    if arch["module_class"] == "FSM":
        fsm = arch.get("fsm_specific", {})
        if not fsm.get("state_encoding") or not fsm.get("state_transitions"):
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
  - NEVER convert assign statements to reg assignments — preserve all assign statements.
  - If the code is clean, return {"status": "PASSED", "fixed_code": ""}.
  - If patching, return {"status": "REJECTED", "fixed_code": "<full corrected module>"}.
Return pure JSON: {"status": "...", "fixed_code": "..."}
"""
    return run_llm(v_code, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)

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
    if previous_ports:
        port_rule = (
            f"2. STRICT INTERFACE LOCK: You MUST reuse this exact port list verbatim:\n"
            f"   {previous_ports}"
        )
    else:
        port_rule = (
            "2. STRICT INTERFACE: Port declarations MUST perfectly match "
            "the Architect's `port_interface` array."
        )

    if previous_error and best_v_code:
        failure_type  = classify_failure(previous_error)
        error_context = f"""
MUTATION REQUIRED — previous RTL failed.

FAILURE CLASSIFICATION: {failure_type}

SIMULATION OUTPUT:
{previous_error}

TARGETED FIX RULES based on failure type:
- INCREMENT_LOGIC  → modify only the increment counting path
- DECREMENT_LOGIC  → modify only the decrement counting path
- OFF_BY_ONE       → check boundary conditions: is the count initialized to 0 or 1?
                     Is the comparison using < vs <= ? Fix only the affected boundary.
- OVERFLOW_FLAG    → fix only the overflow flag assign statement
- UNDERFLOW_FLAG   → fix only the underflow flag assign statement
- RESET_LOGIC      → fix only the reset branch inside always @(posedge clk or negedge rst_n)
- LOAD_LOGIC       → fix only the load_en priority branch
- TB_TIMEOUT       → check for unintended combinational loops or missing $finish paths
- UNKNOWN          → diagnose carefully before changing anything

You MUST ONLY modify logic related to the failure type above.
Do not touch unrelated blocks.

PREVIOUS BEST VERILOG (mutate this, do not rewrite from scratch):
{best_v_code}
"""
    elif previous_error:
        error_context = f"""
SYNTAX ERROR IN PREVIOUS ATTEMPT:
{previous_error}
Fix only the line causing this error. Do not restructure the module.
"""
    else:
        error_context = ""

    # Extract algorithm description from architecture for RTL generator
    control_logic_str = ""
    if architecture.get("control_logic"):
        lines = []
        for cl in architecture["control_logic"]:
            lines.append(f"  Register '{cl.get('register')}': {cl.get('algorithm')}")
        control_logic_str = "REGISTER NEXT-STATE ALGORITHMS (transcribe these exactly):\n" + "\n".join(lines)

    flag_logic_str = ""
    if architecture.get("flag_logic"):
        lines = []
        for fl in architecture["flag_logic"]:
            lines.append(
                f"  Signal '{fl.get('signal')}' [{fl.get('type','combinational')}]: "
                f"assert when {fl.get('condition')}"
            )
        flag_logic_str = "FLAG LOGIC (implement as assign statements for combinational, register for sequential):\n" + "\n".join(lines)

    priority_str = ""
    if architecture.get("priority_order"):
        priority_str = (
            "CONTROL PRIORITY (implement as nested if-else in this exact order, highest first):\n"
            + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(architecture["priority_order"]))
        )

    algorithm_section = "\n\n".join(filter(None, [control_logic_str, flag_logic_str, priority_str]))
    if algorithm_section:
        algorithm_section = f"\nALGORITHM DESCRIPTION — implement by transcription, no invention:\n{algorithm_section}\n"

    system_prompt = f"""
You are an elite RTL design engineer. Generate ONLY the synthesizable Verilog RTL module.
Do NOT generate a testbench — a separate agent handles that.
{error_context}{algorithm_section}
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
10. COMBINATIONAL FLAGS: If the spec says a flag "goes high WHEN condition is true"
    (present tense, no clock mentioned), implement as a continuous assign statement:
    assign flag = (condition using current inputs and state);
    These are wires. Do NOT register them inside always @(posedge clk).

Return pure JSON:
{{
  "structural_reasoning": "...",
  "verilog_code": "..."
}}
"""
    return run_llm(prompt, system_prompt, provider,
                   max_api_retries=2, cand_idx=cand_idx)

# ==============================================================================
# AGENT: TESTBENCH GENERATOR (independent from RTL generator)
# ==============================================================================
def generate_testbench(
    prompt:         str,
    provider:       str,
    design_name:    str,
    v_code:         str,
    previous_error: str = None,
    previous_tb:    str = None,
    cand_idx:       int = 0,
) -> str:
    error_context = ""
    if previous_error and previous_tb:
        failure_type  = classify_failure(previous_error)
        error_context = f"""
PREVIOUS TESTBENCH FAILED — fix the testbench only, the RTL is correct.

FAILURE CLASSIFICATION: {failure_type}

SIMULATION OUTPUT:
{previous_error}

FAILING TESTBENCH:
{previous_tb}

Fix only what caused the failure. Do not change passing test cases.
"""

    system_prompt = f"""
You are a hardware verification engineer. Write a Verilog-2001 testbench.

RTL MODULE TO TEST:
{v_code}

ORIGINAL SPEC:
{prompt}

TESTBENCH RULES — all mandatory:
1. `timescale 1ns/1ps
2. Verilog-2001 only. No SystemVerilog anywhere.
3. Declare ALL variables at top of module, never inside begin/end blocks.
4. Drive DUT inputs through declared reg variables only.
   NEVER write to dut.portname directly.
5. Clock: always #5 clk = ~clk; with initial clk = 0;
6. TIMING: Apply stimulus at negedge clk, sample at the NEXT negedge clk.
   Correct pattern:
     @(negedge clk); // apply inputs here
     @(negedge clk); // sample outputs here — RTL has had a full posedge to register
   NEVER sample at @(posedge clk) immediately after applying stimulus — race condition.
7. NO WHILE LOOPS. Use only bounded repeat() or for loops with a fixed count.
8. GOLDEN MODEL: Implement ONLY what the spec explicitly states.
   Do not add wrap-around, saturation, or any behavior not stated in the spec.
   If spec says a flag "goes high WHEN condition is true" — treat it as combinational,
   check it at the same negedge where you sample count_out, not one cycle later.
9. EVERY code path must call $finish. No path may reach $endmodule without $finish.
10. Success: $display("SIMULATION_SUCCESS"); $finish;
11. Timeout: initial begin #100000; $display("FAIL: Timeout"); $finish; end
{error_context}

Return pure JSON:
{{
  "testbench_code": "..."
}}
"""
    result = run_llm(prompt, system_prompt, provider,
                     max_api_retries=2, cand_idx=cand_idx)
    return clean_code_string(result.get("testbench_code", ""))

# ==============================================================================
# PORT MATCHING
# ==============================================================================
def ports_match(v_code: str, locked_ports_json: str) -> bool:
    if not locked_ports_json:
        return True
    locked_ports = json.loads(locked_ports_json)

    m = re.search(
        r'module\s+\w+\s*(?:#\s*\(.*?\))?\s*\((.*?)\);',
        v_code, re.S
    )
    if not m:
        return False

    port_str     = m.group(1)
    # findall correctly handles bus declarations like [7:0] without false splits
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
        return False, f"Compile failed:\n{truncate_log(cr.stderr)}"

    try:
        sr = subprocess.run(["vvp", output_bin],
                            capture_output=True, text=True, timeout=60)
        if sr.returncode != 0:
            return False, f"Runtime failed:\n{truncate_log(sr.stderr)}"
        if any(kw in sr.stdout for kw in ("ERROR", "Error", "FAIL", "Fail")):
            return False, f"Verification failed:\n{truncate_log(sr.stdout, 15)}"
        if "SIMULATION_SUCCESS" not in sr.stdout or "finish" not in sr.stdout.lower():
            return False, "Missing SIMULATION_SUCCESS — simulation exited prematurely or timed out."
        return True, "Simulation clean."
    except subprocess.TimeoutExpired:
        return False, "Simulation timeout (possible infinite loop)."

def verify_with_yosys(v_file: str):
    cmd = ["yosys", "-p",
           f"read_verilog {v_file}; proc; opt; fsm; memory; check -assert; prep; stat"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        log = r.stderr + "\n" + r.stdout
        if r.returncode == 0 and "ERROR" not in log:
            dffs  = len(re.findall(r'\$(?:a?dff|dffe|sdff)', log, re.IGNORECASE))
            cells = re.search(r'Number of cells:\s+(\d+)', log)
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
        print("   ❌ Clarifier returned empty spec. Check your API key.")
        return False

    arch_hashes           = set()
    code_hashes           = set()
    locked_ports          = None
    error_log             = None
    architecture          = None
    architecture_flawed   = True

    best_score            = -1
    best_v_code           = None
    best_tb_code          = None
    best_err_log          = None
    global_candidate_id   = 0
    consecutive_sim_fails = 0

    for attempt in range(max_retries):
        print(f"\n{'─'*60}")
        print(f"🔄 Generation {attempt + 1}/{max_retries}")
        print(f"{'─'*60}")

        # Reset after 5 consecutive sim failures (increased from 3)
        if consecutive_sim_fails >= 5:
            print("   🔄 5 consecutive sim failures — forcing full architecture redesign...")
            architecture_flawed   = True
            consecutive_sim_fails = 0
            best_score            = -1
            best_v_code           = None
            best_tb_code          = None
            best_err_log          = None
            locked_ports          = None
            error_log             = "CRITICAL: Previous architecture failed repeatedly. Produce a structurally different blueprint."

        if architecture_flawed:
            architecture = generate_architecture(clarified_spec, provider, error_log, attempt)
            arch_hash    = code_hash(json.dumps(architecture, sort_keys=True))

            if not validate_architecture(architecture):
                port_count = len(architecture.get("port_interface", []))
                print(f"   ⚠️  Architect returned invalid blueprint (ports={port_count}). Retrying...")
                error_log = (
                    f"CRITICAL: Blueprint has only {port_count} ports. "
                    f"You MUST list ALL ports from the spec in port_interface. Retry with complete array."
                )
                continue

            if arch_hash in arch_hashes:
                print("   ⚠️  Architect produced a duplicate blueprint. Forcing variation...")
                error_log = (
                    "CRITICAL: Do not repeat previous blueprints. "
                    "Change register decomposition, rename signals, or restructure the datapath."
                )
                continue

            arch_hashes.add(arch_hash)
            architecture_flawed = False
            print(f"   📐 Architect: New blueprint accepted (class={architecture.get('module_class', '?')})")
        else:
            print("   📐 Architect: Reusing stable blueprint.")

        # ── RTL generation (parallel) ─────────────────────────────────────────
        if best_v_code:
            print(f"   🧬 Evolution Engine: Mutating best RTL (score={best_score}/2)...")
        else:
            print(f"   🛠️  Generator: Spawning {candidates_per_round} parallel RTL candidates...")

        rtl_candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidates_per_round) as executor:
            futures = {}
            for i in range(candidates_per_round):
                time.sleep(2)
                futures[executor.submit(
                    generate_rtl,
                    base_prompt, provider, design_name, architecture,
                    locked_ports,
                    best_err_log or error_log,
                    best_v_code,
                    i
                )] = i

            try:
                for future in concurrent.futures.as_completed(futures, timeout=420):
                    try:
                        result = future.result(timeout=200)
                    except concurrent.futures.TimeoutError:
                        print("   ⚠️  A thread timed out individually — skipping.")
                        continue
                    except Exception as e:
                        print(f"   ⚠️  Thread raised exception: {e}")
                        continue

                    v_code = clean_code_string(result.get("verilog_code", ""))
                    if len(v_code) < 50:
                        print("   ⚠️  Candidate returned empty Verilog. Discarding.")
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
                if fixed:
                    ratio             = len(fixed) / len(v_code)
                    has_assign_before = bool(re.search(r'\bassign\b', v_code))
                    has_assign_after  = bool(re.search(r'\bassign\b', fixed))
                    if has_assign_before and not has_assign_after:
                        print(f"      ⚠️  Candidate {idx+1}: Reviewer removed assign statements. Rejecting patch.")
                        reviewed_candidates.append(v_code)
                    elif 0.75 < ratio < 1.3:
                        print(f"      ✅ Candidate {idx+1}: Patch applied.")
                        reviewed_candidates.append(fixed)
                    else:
                        print(f"      ⚠️  Candidate {idx+1}: Hallucinated rewrite (ratio={ratio:.2f}). Keeping original.")
                        reviewed_candidates.append(v_code)
                else:
                    reviewed_candidates.append(v_code)
            else:
                reviewed_candidates.append(v_code)

        # Lock port interface after first successful RTL generation
        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture["port_interface"])
            print(f"   🔒 Port interface locked ({len(architecture['port_interface'])} ports).")

        # ── Testbench generation ──────────────────────────────────────────────
        # TB LOCKING: If the previous failure was a logic mismatch (not a TB
        # structural problem), the testbench was working correctly as a fitness
        # function. Reuse it rather than mutating it — mutating a working
        # testbench to accommodate broken RTL is counterproductive.
        tb_is_logic_failure = (
            best_tb_code is not None
            and best_err_log is not None
            and classify_failure(best_err_log) not in ("TB_TIMEOUT", "SYNTAX_ERROR", "UNKNOWN")
        )

        print(f"   📋 Testbench Agent: ", end="")
        if tb_is_logic_failure:
            print(f"Reusing locked testbench (logic failure — TB is correct fitness function).")
        else:
            print(f"Generating {len(reviewed_candidates)} testbench(es) in parallel...")

        def _gen_tb(args):
            idx, v_code = args
            if tb_is_logic_failure:
                # Reuse the locked testbench — don't regenerate
                return idx, v_code, best_tb_code
            tb = generate_testbench(
                base_prompt, provider, design_name, v_code,
                previous_error=best_err_log if best_tb_code else None,
                previous_tb=best_tb_code,
                cand_idx=idx,
            )
            return idx, v_code, tb

        candidates = []  # list of (v_code, tb_code) tuples
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(reviewed_candidates)) as tb_executor:
            tb_futures = [
                tb_executor.submit(_gen_tb, (idx, v_code))
                for idx, v_code in enumerate(reviewed_candidates)
            ]
            for tb_future in concurrent.futures.as_completed(tb_futures, timeout=300):
                try:
                    idx, v_code, tb_code = tb_future.result(timeout=200)
                except Exception as e:
                    print(f"      ⚠️  Testbench generation failed: {e}")
                    continue
                if not tb_code or len(tb_code) < 50:
                    print(f"      ⚠️  Testbench {idx+1}: Empty. Discarding candidate.")
                    continue
                status = "🔒 Locked" if tb_is_logic_failure else "✅ Generated"
                print(f"      {status} Testbench {idx+1}.")
                candidates.append((v_code, tb_code))

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
            score = 0

            passed, log = verify_with_verilator(v_file, design_name, v_code)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Verilator: {err}")
                if score > best_score:
                    best_score   = score
                    best_err_log = f"Syntax Error:\n{err}"
                continue

            score = 1
            print("      ✅ Verilator passed.")

            passed, log = verify_with_iverilog(v_file, tb_file, design_name, global_candidate_id)
            if not passed:
                err              = extract_primary_error(log)
                full_sim_context = truncate_log(log, 30)
                failure_type     = classify_failure(full_sim_context)
                print(f"      ❌ Icarus [{failure_type}]: {err}")
                if score > best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_tb_code = tb_code
                    best_err_log = (
                        f"FAILURE TYPE: {failure_type}\n"
                        f"Simulation Error (full output):\n{full_sim_context}"
                    )
                continue

            score = 2
            round_sim_passed = True
            print("      ✅ Icarus simulation passed.")

            passed, log = verify_with_yosys(v_file)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Yosys: {err}")
                if score > best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_err_log = f"Synthesis Error:\n{err}"
                continue

            if (re.search(r"DFFs:\s*0", log, re.IGNORECASE)
                    and any(kw in base_prompt.lower() for kw in ("clk", "clock", "sync", "sequential"))):
                print("      ⚠️  Yosys: 0 DFFs synthesized for a sequential design.")
                if score > best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_err_log = (
                        "CRITICAL: Sequential design produced 0 registers. "
                        "Ensure all state registers use non-blocking assignments "
                        "inside always @(posedge clk)."
                    )
                continue

            print(f"      ✅ Yosys passed: {log}")
            round_passed = True
            break

        if round_passed:
            print(f"\n{'═'*60}")
            print("🎉 PIPELINE COMPLETE — Structural + Functional + Physical verification passed.")
            print(f"📁 Winning design: ./workspace/{design_name}/candidate_{global_candidate_id}/")
            print(f"{'═'*60}\n")
            return True

        if best_score == 1 and not round_sim_passed:
            consecutive_sim_fails += 1
        else:
            consecutive_sim_fails = 0

        print(f"\n   📊 Round summary: best_score={best_score}/2 | "
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
