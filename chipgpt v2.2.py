import os
import json
import argparse
import subprocess
import hashlib
import re
import time
import concurrent.futures
from openai import OpenAI
from groq import Groq

# ==============================================================================
# MODEL TIER CONFIGURATION
# ------------------------------------------------------------------------------
FAST_MODEL   = "gpt-4o-mini"
STRONG_MODEL = "gpt-5-mini"

# Reasoning model families — controls whether we use temperature or reasoning_effort
REASONING_MODELS = {
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "gpt-5-mini", "gpt-5",          # add new ones here as released
}

def _is_reasoning_model(model_name: str) -> bool:
    return model_name.strip() in REASONING_MODELS

# Max tokens per model tier
MAX_TOKENS_FAST   = 4096
MAX_TOKENS_STRONG = 16384

# ==============================================================================
# GLOBAL CLIENT (Lazy Initialization)
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
        # Fix invalid backslash escapes from Verilog embedded in JSON strings
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

    model_name = FAST_MODEL if force_fast_model else STRONG_MODEL
    is_reasoning = _is_reasoning_model(model_name)
    max_tokens = MAX_TOKENS_FAST if force_fast_model else MAX_TOKENS_STRONG

    for attempt in range(max_api_retries):
        try:
            if provider == "openai":
                client = get_openai_client()

                kwargs = {
                    "model":      model_name,
                    "messages":   [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "timeout": 180 if not force_fast_model else 90,
                }

                if is_reasoning:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens

                if not is_reasoning:
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["temperature"]      = temps[(cand_idx + attempt) % 3] # Dynamic Temp Shift
                else:
                    kwargs["reasoning_effort"] = efforts[(cand_idx + attempt) % 3]

                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: "
                          f"Calling {model_name} (Attempt {attempt+1})...")

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

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
# AGENT DEFINITIONS
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


def generate_architecture(spec: dict, provider: str, previous_error: str = None, attempt_idx: int = 0) -> dict:
    print(f"   📐 Routing to ARCHITECT AGENT ({provider.upper()}) [{FAST_MODEL}]...")
    error_context = (
        f"\nCRITICAL FEEDBACK FROM PREVIOUS RUN:\n{previous_error}\n"
        if previous_error else ""
    )
    system_prompt = f"""
You are an elite SoC Architect. Design a strictly synthesizable hardware architecture.
{error_context}

CRITICAL: Classify the design correctly:
- DATAPATH: counters, FIFOs, ALUs, datapaths, memories, shift registers, arithmetic units.
- FSM: protocol controllers, sequence detectors, handshake controllers, state machines.
A counter with overflow/underflow flags is always DATAPATH, never FSM.

Return pure JSON exactly matching this schema:
{{
    "module_class": "FSM" or "DATAPATH",
    "clock_and_reset": {{"clock": "clk", "reset": "rst_n", "active_low": true}},
    "port_interface": [{{"name": "clk", "direction": "input", "width": 1}}],
    "registers": [{{"name": "...", "width": 8}}],
    "memory_blocks": [{{"name": "...", "width": 8, "depth": 16}}],
    "datapath_signals": [{{"name": "...", "width": 8}}],
    "fsm_specific": {{"state_encoding": {{}}, "state_transitions": []}}
}}

[VARIATION SALT: {time.time()} - Ensure uniqueness if retrying]
"""
    return run_llm(json.dumps(spec), system_prompt, provider,
                   cand_idx=attempt_idx, force_fast_model=True, silent=True)


def validate_architecture(arch: dict) -> bool:
    if not arch or "module_class" not in arch:
        return False
    if not arch.get("port_interface"):
        return False
    if arch["module_class"] == "FSM":
        fsm = arch.get("fsm_specific", {})
        if not fsm.get("state_encoding") or not fsm.get("state_transitions"):
            return False
    return True


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
    return run_llm(v_code, system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)


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
    declarations = [p.strip() for p in re.split(r',', port_str)]

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


def generate_hardware(
    prompt:         str,
    provider:       str,
    design_name:    str,
    architecture:   dict,
    previous_ports: str  = None,
    previous_error: str  = None,
    best_v_code:    str  = None,
    best_tb_code:   str  = None,
    cand_idx:       int  = 0,
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
        tb_context = f"\nFAILING TESTBENCH (read this to understand the stimulus):\n{best_tb_code}" if best_tb_code else ""
        error_context = f"""
MUTATION REQUIRED — previous design failed simulation.

SIMULATION OUTPUT:
{previous_error}
{tb_context}

DIAGNOSIS STEPS (do these before writing any code):
1. Read the simulation output and testbench together. Find which stimulus triggered the failure.
2. Trace through the RTL to find the exact assignment responsible for the wrong output.
3. Fix only those lines. Do not restructure the module.

PREVIOUS BEST VERILOG (mutate this, do not rewrite from scratch):
{best_v_code}
"""
    elif previous_error:
        error_context = f"""
=== SYNTAX ERROR IN PREVIOUS ATTEMPT ===
ERROR: {previous_error}
Identify the exact line causing this and fix it. Do not restructure the module.
"""
    else:
        error_context = ""

    system_prompt = f"""
You are an elite RTL design engineer.
{error_context}
Implement exactly the architecture below designed by the Senior Architect:
{json.dumps(architecture, indent=2)}

SILICON RULES — all mandatory:
1. Top module named exactly: `{design_name}`
{port_rule}
3. Verilog-2001 only. No SystemVerilog. No initial blocks in synthesizable modules.
4. Sequential always @(posedge clk): non-blocking (<=) only.
   Combinational always @(*): blocking (=) only.
5. No latches: assign defaults to all signals at top of every always @(*) block.
6. Parameters: module {design_name} #(parameter WIDTH=8) (input clk, ...);
7. No expression part-selects: wire [N:0] tmp = a-b; use tmp[3:0], not (a-b)[3:0].
8. ASYNC RESET: active-low means the reset condition is when the signal is LOW (0).
   Sensitivity list: always @(posedge clk or negedge rst_n)
   Reset check: if (!rst_n) begin <reset values here> end else begin <normal logic> end
9. SINGLE DRIVER: Every reg must be driven by exactly ONE always block.
10. OUTPUT FLAG TIMING: Flags like overflow, underflow, full, empty, valid MUST be
    registered in the same always block as the data they describe, OR purely combinational via assign.

TESTBENCH RULES (CRITICAL FOR PASSING):
- `timescale 1ns/1ps`
- Verilog-2001 only. No SystemVerilog anywhere in the testbench.
- NO INFINITE LOOPS: NEVER use `while` loops or unbounded `wait` statements.
- USE FOR LOOPS: Only use strictly bounded `for` loops (e.g., `for(i=0; i<300; i=i+1)`).
- TIMING (CRITICAL): After applying stimulus at `negedge clk`, always wait one FULL clock period before sampling outputs.
  The correct pattern is:
    @(negedge clk); // apply inputs
    @(negedge clk); // wait for a full posedge update, then sample outputs
  NEVER sample at @(posedge clk) immediately after applying stimulus at the preceding negedge — this creates a race condition!
- GOLDEN MODEL: The golden model must implement ONLY what the spec says. Do not add behaviors (like wrap-around) that the spec does not explicitly state. 
- COMBINATIONAL FLAGS: Check them AFTER the clock edge that would trigger them, not at the exact same edge.
- Timeout: initial begin #10000; $display("FAIL: Timeout"); $finish; end
- Success: $display("SIMULATION_SUCCESS"); $finish; — print this only after ALL tests pass.

Return pure JSON:
{{
  "structural_reasoning": "...",
  "implemented_fsm_states": [],
  "implemented_datapath_blocks": [],
  "signal_table": [],
  "verilog_code": "...",
  "testbench_code": "..."
}}
"""

    design = run_llm(prompt, system_prompt, provider,
                     max_api_retries=2,
                     cand_idx=cand_idx)
    if not design:
        return {}

    return design

# ==============================================================================
# WORKSPACE + VERIFICATION
# ==============================================================================
def save_to_workspace(design_name: str, design: dict, candidate_id: int):
    workspace = f"./workspace/{design_name}/candidate_{candidate_id}"
    os.makedirs(f"{workspace}/src", exist_ok=True)
    os.makedirs(f"{workspace}/tb",  exist_ok=True)

    v_code  = clean_code_string(design.get("verilog_code",  ""))
    tb_code = clean_code_string(design.get("testbench_code", ""))

    v_file  = f"{workspace}/src/{design_name}.v"
    tb_file = f"{workspace}/tb/{design_name}_tb.v"

    with open(v_file,  "w") as f: f.write(v_code)
    with open(tb_file, "w") as f: f.write(tb_code)
    return v_file, tb_file, v_code

def verify_with_verilator(v_file: str, design_name: str, v_code: str):
    if not re.search(r'\bmodule\s+' + re.escape(design_name) + r'\b', v_code):
        return False, f"%Error: Top module '{design_name}' not found."
    cmd = ["verilator", "--lint-only", "-Wall",
       "-Wno-DECLFILENAME",
       "-Wno-UNUSED",
       "-Wno-UNDRIVEN", 
       "-Werror-WIDTH",
       "-Werror-IMPLICIT",
       v_file]
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

        if consecutive_sim_fails >= 3:
            print("   🔄 3 consecutive sim failures — forcing full architecture redesign...")
            architecture_flawed   = True
            consecutive_sim_fails = 0
            best_score            = -1
            best_v_code           = None
            best_tb_code          = None
            best_err_log          = None
            locked_ports          = None   
            error_log             = f"CRITICAL: The previous architecture hash completely failed simulation. You MUST output a structurally different JSON blueprint."   

        if architecture_flawed:
            architecture = generate_architecture(clarified_spec, provider, error_log, attempt)
            arch_hash    = code_hash(json.dumps(architecture, sort_keys=True))

            if not validate_architecture(architecture):
                print("   ⚠️  Architect returned invalid blueprint. Retrying...")
                error_log = "CRITICAL: Blueprint is missing required fields. Retry with complete schema."
                continue

            if arch_hash in arch_hashes:
                print("   ⚠️  Architect produced a duplicate blueprint. Forcing variation...")
                error_log = "CRITICAL: Do not repeat previous blueprints. Change the module_class, or add/rename a register in datapath_signals to force variation."
                continue

            arch_hashes.add(arch_hash)
            architecture_flawed = False
            print(f"   📐 Architect: New blueprint accepted "
                  f"(class={architecture.get('module_class', '?')})")
        else:
            print("   📐 Architect: Reusing stable blueprint.")

        if best_v_code:
            print(f"   🧬 Evolution Engine: Mutating best candidate (score={best_score}/2)...")
        else:
            print(f"   🛠️  Generator: Spawning {candidates_per_round} parallel candidates...")

        candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidates_per_round) as executor:
            futures = {}
            for i in range(candidates_per_round):
                time.sleep(2)  # stagger launches to avoid RPM burst
                futures[executor.submit(
                    generate_hardware,
                    base_prompt, provider, design_name, architecture,
                    locked_ports,
                    best_err_log or error_log,
                    best_v_code,
                    best_tb_code,
                    i
                )] = i

            try:
                for future in concurrent.futures.as_completed(futures, timeout=420):
                    design = future.result()
                    v_code = clean_code_string(design.get("verilog_code", ""))

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
                            and not re.search(
                                r'reg\s*\[[^\]]+\]\s*\w+\s*\[[^\]]+\]', v_code)):
                        print("   ⚠️  DATAPATH with memory_blocks missing 2D reg array. Discarding.")
                        continue

                    candidates.append(design)

            except concurrent.futures.TimeoutError:
                print("   ⚠️  Executor timeout — API likely hung. Restarting round.")

        if not candidates:
            print("   ❌ No valid candidates this round.")
            error_log = "All candidates were empty, had wrong ports, or were duplicates."
            continue

        # Serial reviewer pass — runs after all threads have completed
        print(f"   🔎 Reviewer: Scanning {len(candidates)} candidate(s) for latches/loops...")
        for idx, design in enumerate(candidates):
            v_code = clean_code_string(design.get("verilog_code", ""))
            if len(v_code) > 50:
                review = review_hardware(v_code, provider)
                if review.get("status") == "REJECTED":
                    fixed = clean_code_string(review.get("fixed_code", ""))
                    if fixed and len(v_code) > 0:
                        ratio = len(fixed) / len(v_code)
                        if 0.75 < ratio < 1.3:
                            print(f"      ✅ Candidate {idx+1}: Patch applied.")
                            candidates[idx]["verilog_code"] = fixed
                        else:
                            print(f"      ⚠️  Candidate {idx+1}: Hallucinated rewrite (ratio={ratio:.2f}). Discarding.")

        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture["port_interface"])
            print(f"   🔒 Port interface locked ({len(architecture['port_interface'])} ports).")

        round_passed     = False
        round_sim_passed = False

        for design in candidates:
            global_candidate_id += 1
            print(f"\n   🔬 Testing Candidate #{global_candidate_id}...")

            v_file, tb_file, v_code = save_to_workspace(
                design_name, design, global_candidate_id)
            score = 0

            passed, log = verify_with_verilator(v_file, design_name, v_code)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Verilator: {err}")
                if score >= best_score:
                    best_score   = score
                    best_err_log = f"Syntax Error:\n{err}"
                continue

            score = 1
            print("      ✅ Verilator passed.")

            passed, log = verify_with_iverilog(v_file, tb_file, design_name, global_candidate_id)
            if not passed:
                err = extract_primary_error(log)
                full_sim_context = truncate_log(log, 30)
                print(f"      ❌ Icarus: {err}")
                if score >= best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_tb_code = clean_code_string(design.get("testbench_code", ""))
                    best_err_log = f"Simulation Error (full output):\n{full_sim_context}"
                continue

            score = 2
            round_sim_passed = True
            print("      ✅ Icarus simulation passed.")

            passed, log = verify_with_yosys(v_file)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Yosys: {err}")
                if score >= best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_err_log = f"Synthesis Error:\n{err}"
                continue

            if (re.search(r"DFFs:\s*0", log, re.IGNORECASE)
                    and any(kw in base_prompt.lower() for kw in ("clk", "clock", "sync", "sequential"))):
                print("      ⚠️  Yosys: 0 DFFs synthesized for a sequential design.")
                if score >= best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_err_log = ("CRITICAL: Sequential design produced 0 registers. "
                                    "Ensure all state registers use non-blocking assignments "
                                    "inside always @(posedge clk).")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ChipGPT: Autonomous RTL Generator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "prompt", type=str,
        help="Natural language description of the hardware to generate."
    )
    parser.add_argument(
        "--name", type=str, default="my_module",
        help="Exact name of the top-level Verilog module (default: my_module)."
    )
    parser.add_argument(
        "--provider", type=str, choices=["openai", "groq"], default="openai",
        help="LLM provider to use (default: openai)."
    )
    parser.add_argument(
        "--retries", type=int, default=10,
        help="Maximum number of generation attempts (default: 10)."
    )
    parser.add_argument(
        "--candidates", type=int, default=3,
        help="Parallel candidates per round (default: 3)."
    )
    args = parser.parse_args()

    success = autonomous_build_loop(
        base_prompt=args.prompt,
        design_name=args.name,
        provider=args.provider,
        max_retries=args.retries,
        candidates_per_round=args.candidates,
    )
    exit(0 if success else 1)
