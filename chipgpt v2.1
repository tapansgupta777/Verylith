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
# Two-tier model strategy:
#   FAST_MODEL  → Clarifier, Architect, Reviewer (low complexity, high volume)
#   STRONG_MODEL → Generator (high complexity, needs reasoning capability)
#
# To switch models, change only these two constants. Nothing else needs to change.
#
# Confirmed working values:
#   FAST_MODEL   = "gpt-4o-mini"
#   STRONG_MODEL = "gpt-4o"          ← safe baseline, widely available
#   STRONG_MODEL = "gpt-5-mini"      ← use this when you have access
#   STRONG_MODEL = "o3-mini"         ← reasoning model, also good
#
# NOTE: If STRONG_MODEL is a reasoning model (o1/o3/gpt-5 family), the script
# automatically switches from temperature to reasoning_effort. No code changes needed.
# ==============================================================================

FAST_MODEL   = "gpt-4o-mini"
STRONG_MODEL = "gpt-5-mini"

# Reasoning model families — controls whether we use temperature or reasoning_effort
REASONING_MODEL_PREFIXES = ("o1", "o3", "gpt-5")

def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in REASONING_MODEL_PREFIXES)

# Max tokens per model tier — prevents silent truncation of large Verilog+TB responses
MAX_TOKENS_FAST   = 4096
MAX_TOKENS_STRONG = 8192

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
    """Return the first error line with ±1 lines of context."""
    lines = log.strip().split('\n')
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ("error", "mismatch", "fail", "syntax")):
            start = max(0, i - 1)
            end   = min(len(lines), i + 2)
            return "\n".join(lines[start:end])
    return truncate_log(log, 5)

def parse_llm_json(raw_text: str) -> dict:
    """
    Robust JSON extractor. Handles:
      - Markdown code fences (```json ... ```)
      - Leading/trailing prose around the JSON object
      - Truncated responses (returns {} so callers can detect failure)
    """
    try:
        triple_tick = "`" * 3
        clean = re.sub(
            rf'{triple_tick}(?:json)?\s*|{triple_tick}\s*$', '',
            raw_text.strip(), flags=re.MULTILINE
        ).strip()
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group(), strict=False)
        return json.loads(clean, strict=False)
    except json.JSONDecodeError as e:
        print(f"   ⚠️  JSON Parse Error: {e}")
        return {}

def clean_code_string(raw_code: str) -> str:
    """Strip markdown fences from LLM-generated Verilog."""
    if not raw_code:
        return ""
    clean = raw_code.strip()
    for fence in ("```verilog", "```systemverilog", "```v", "```"):
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
    """
    Unified LLM call with:
      - Two-tier model selection (FAST vs STRONG)
      - Auto-detection of reasoning models (effort vs temperature)
      - Explicit max_tokens to prevent silent truncation
      - Exponential-style backoff on rate limits
    """
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
                    "max_tokens": max_tokens,
                    "timeout":    180 if not force_fast_model else 90,
                }

                # Reasoning models do not support response_format or temperature
                if not is_reasoning:
                    kwargs["response_format"] = {"type": "json_object"}
                    kwargs["temperature"]      = temps[cand_idx % 3]
                else:
                    kwargs["reasoning_effort"] = efforts[cand_idx % 3]

                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: "
                          f"Calling {model_name} (Attempt {attempt+1})...")

                response = client.chat.completions.create(**kwargs)
                return parse_llm_json(response.choices[0].message.content)

            elif provider == "groq":
                client = Groq()
                if not silent:
                    print(f"      📡 [API] Thread {cand_idx+1}: Calling Groq llama-3.3-70b...")
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=temps[cand_idx % 3],
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
            if "429" in err or "quota" in err:
                wait = 35 * (attempt + 1)          # escalating backoff
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


def generate_architecture(spec: dict, provider: str, previous_error: str = None) -> dict:
    print(f"   📐 Routing to ARCHITECT AGENT ({provider.upper()}) [{FAST_MODEL}]...")
    error_context = (
        f"\nCRITICAL FEEDBACK FROM PREVIOUS RUN:\n{previous_error}\n"
        if previous_error else ""
    )
    system_prompt = f"""
You are an elite SoC Architect. Design a strictly synthesizable hardware architecture.
{error_context}

CRITICAL: Decide if this is primarily an FSM or a DATAPATH.
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
"""
    return run_llm(json.dumps(spec), system_prompt, provider,
                   cand_idx=0, force_fast_model=True, silent=True)


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
    """
    Pre-flight latch/loop reviewer. Runs FAST_MODEL (cheap, silent).
    Returns {"status": "PASSED"} or {"status": "REJECTED", "fixed_code": "..."}
    Diff guard in generate_hardware prevents hallucinated rewrites.
    """
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
    """
    Validates that generated Verilog preserves the locked port interface.
    FIX 1: Handles parametrized modules with #(parameter...) correctly.
    """
    if not locked_ports_json:
        return True
    locked_ports = json.loads(locked_ports_json)

    # Handles both plain and parametrized module headers
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
    cand_idx:       int  = 0,
) -> dict:
    """
    Core RTL + Testbench generator.
    Uses STRONG_MODEL. Includes post-generation reviewer pass.
    """
    # Port constraint rule
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

    # Evolutionary mutation context
    if previous_error and best_v_code:
        error_context = f"""
=== EVOLUTIONARY MUTATION REQUIRED ===
Your previous best design failed. ERROR:
{previous_error}

PREVIOUS BEST VERILOG (mutate this — do NOT rewrite from scratch):
{best_v_code}

MUTATION RULE: Only modify the lines directly responsible for the error above.
Preserve all correct logic. Changing unrelated lines will introduce new bugs.
======================================
"""
    elif previous_error:
        error_context = f"""
=== SYNTAX ERROR IN PREVIOUS ATTEMPT ===
ERROR: {previous_error}
Identify the exact line causing this and fix it. Do not restructure the module.
========================================
"""
    else:
        error_context = ""

    system_prompt = f"""
You are an elite RTL design engineer.
{error_context}
Implement exactly the architecture below designed by the Senior Architect:
{json.dumps(architecture, indent=2)}

═══════════════════════════════════════════
CRITICAL SILICON RULES (non-negotiable)
═══════════════════════════════════════════
1. MODULE NAME: Top module MUST be named exactly: `{design_name}`
{port_rule}
3. VERILOG-2001 ONLY: No SystemVerilog keywords (logic, always_ff, always_comb).
   No `initial` blocks inside synthesizable modules.
4. BLOCKING / NON-BLOCKING:
   - Sequential always @(posedge clk): use NON-BLOCKING assignments (<=)
   - Combinational always @(*): use BLOCKING assignments (=)
5. NO LATCHES: Every combinational always @(*) block MUST assign default
   values to ALL driven signals at the very top of the block.
6. PARAMETERS: Use Verilog-2001 parameter syntax in the module header if needed:
   module {design_name} #(parameter WIDTH=8, parameter DEPTH=16) (input clk, ...);

═══════════════════════════════════════════
TESTBENCH RULES
═══════════════════════════════════════════
- Begin with `timescale 1ns/1ps
- GOLDEN REFERENCE MODEL: Use a fundamentally DIFFERENT algorithm than the RTL.
  Examples:
    · If RTL uses pointer arithmetic  → golden model uses a SystemVerilog queue or $queue
    · If RTL uses an FSM              → golden model uses behavioral if/else with a state variable
    · If RTL uses a counter           → golden model uses an integer accumulator
  PURPOSE: The goal is to CATCH bugs, not to confirm them. A golden model that
  shares implementation strategy with the DUT will pass broken RTL.
- TIMEOUT WATCHDOG: `initial begin #100000; $display("FAIL: Timeout"); $finish; end`
- PASS CONDITION: At the very end of a successful run, print EXACTLY:
  `$display("SIMULATION_SUCCESS");` then call `$finish;`
- Test ALL boundary conditions: empty read, full write, simultaneous rd+wr, reset mid-operation.

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

    design = run_llm(prompt, system_prompt, provider, cand_idx=cand_idx)
    if not design:
        return {}

    # ── Pre-flight Reviewer Pass ───────────────────────────────────────────────
    v_code = clean_code_string(design.get("verilog_code", ""))
    if len(v_code) > 50:
        print(f"      🔎 [Reviewer] Thread {cand_idx+1}: Scanning for latches/loops...")
        review = review_hardware(v_code, provider)
        if review.get("status") == "REJECTED":
            fixed = clean_code_string(review.get("fixed_code", ""))
            if fixed and len(v_code) > 0:
                ratio = len(fixed) / len(v_code)
                if 0.75 < ratio < 1.3:
                    # Safe patch — size is close to original
                    print(f"      ✅ [Reviewer] Thread {cand_idx+1}: Patch applied.")
                    design["verilog_code"] = fixed
                else:
                    # Reviewer tried to rewrite the whole thing — discard
                    print(f"      ⚠️  [Reviewer] Thread {cand_idx+1}: "
                          f"Hallucinated rewrite (ratio={ratio:.2f}). Discarding.")

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
    cmd = ["verilator", "--lint-only", "-Wall", "-Wno-DECLFILENAME",
           "-Werror-WIDTH", v_file]
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
                            capture_output=True, text=True, timeout=20)
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

    # ── State ─────────────────────────────────────────────────────────────────
    arch_hashes           = set()
    code_hashes           = set()
    locked_ports          = None
    error_log             = None
    architecture          = None
    architecture_flawed   = True

    best_score            = -1
    best_v_code           = None
    best_err_log          = None
    global_candidate_id   = 0
    consecutive_sim_fails = 0

    for attempt in range(max_retries):
        print(f"\n{'─'*60}")
        print(f"🔄 Generation {attempt + 1}/{max_retries}")
        print(f"{'─'*60}")

        # ── Architecture reset on persistent simulation failures ───────────────
        # FIX 2: Clear ALL stale state including locked_ports and error_log
        if consecutive_sim_fails >= 3:
            print("   🔄 3 consecutive sim failures — forcing full architecture redesign...")
            architecture_flawed   = True
            consecutive_sim_fails = 0
            best_score            = -1
            best_v_code           = None
            best_err_log          = None
            locked_ports          = None   # prevents stale ports from poisoning new arch
            error_log             = None   # architect starts completely fresh

        # ── Architecture generation ────────────────────────────────────────────
        if architecture_flawed:
            architecture = generate_architecture(clarified_spec, provider, error_log)
            arch_hash    = code_hash(json.dumps(architecture, sort_keys=True))

            if not validate_architecture(architecture):
                print("   ⚠️  Architect returned invalid blueprint. Retrying...")
                error_log = "CRITICAL: Blueprint is missing required fields. Retry with complete schema."
                continue

            if arch_hash in arch_hashes:
                print("   ⚠️  Architect produced a duplicate blueprint. Forcing variation...")
                error_log = "CRITICAL: Do not repeat previous blueprints. Use a different structural approach."
                continue

            arch_hashes.add(arch_hash)
            architecture_flawed = False
            print(f"   📐 Architect: New blueprint accepted "
                  f"(class={architecture.get('module_class', '?')})")
        else:
            print("   📐 Architect: Reusing stable blueprint.")

        # ── Parallel candidate generation ──────────────────────────────────────
        if best_v_code:
            print(f"   🧬 Evolution Engine: Mutating best candidate (score={best_score}/2)...")
        else:
            print(f"   🛠️  Generator: Spawning {candidates_per_round} parallel candidates...")

        candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidates_per_round) as executor:
            futures = {
                executor.submit(
                    generate_hardware,
                    base_prompt, provider, design_name, architecture,
                    locked_ports,
                    best_err_log or error_log,
                    best_v_code,
                    i
                ): i
                for i in range(candidates_per_round)
            }

            try:
                for future in concurrent.futures.as_completed(futures, timeout=240):
                    design = future.result()
                    v_code = clean_code_string(design.get("verilog_code", ""))

                    if len(v_code) < 50:
                        print("   ⚠️  Candidate returned empty Verilog. Discarding.")
                        continue

                    # Port interface lock check
                    if locked_ports and not ports_match(v_code, locked_ports):
                        print("   ⚠️  Port interface mismatch — candidate discarded.")
                        continue

                    # Deduplication
                    h = code_hash(v_code)
                    if h in code_hashes:
                        print("   ⚠️  Duplicate candidate — discarded.")
                        continue
                    code_hashes.add(h)

                    # FIX 3: Memory array check gated on architecture having memory blocks
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

        # Lock ports from first successful architecture generation
        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture["port_interface"])
            print(f"   🔒 Port interface locked ({len(architecture['port_interface'])} ports).")

        # ── Verification pipeline ──────────────────────────────────────────────
        round_passed    = False
        round_sim_passed = False

        for design in candidates:
            global_candidate_id += 1
            print(f"\n   🔬 Testing Candidate #{global_candidate_id}...")

            v_file, tb_file, v_code = save_to_workspace(
                design_name, design, global_candidate_id)
            score = 0

            # Stage 1: Verilator lint
            passed, log = verify_with_verilator(v_file, design_name, v_code)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Verilator: {err}")
                # FIX 4: Strict > (not >=) prevents lower-score candidates
                #         from overwriting better error context
                if score > best_score:
                    best_score   = score
                    best_err_log = f"Syntax Error:\n{err}"
                continue

            score = 1
            print("      ✅ Verilator passed.")

            # Stage 2: Icarus Verilog simulation
            passed, log = verify_with_iverilog(v_file, tb_file, design_name, global_candidate_id)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Icarus: {err}")
                if score > best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_err_log = f"Simulation Error:\n{err}"
                continue

            score = 2
            round_sim_passed = True
            print("      ✅ Icarus simulation passed.")

            # Stage 3: Yosys synthesis
            passed, log = verify_with_yosys(v_file)
            if not passed:
                err = extract_primary_error(log)
                print(f"      ❌ Yosys: {err}")
                if score > best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_err_log = f"Synthesis Error:\n{err}"
                continue

            # Sanity check: sequential design should have DFFs
            if (re.search(r"DFFs:\s*0", log, re.IGNORECASE)
                    and any(kw in base_prompt.lower() for kw in ("clk", "clock", "sync", "sequential"))):
                print("      ⚠️  Yosys: 0 DFFs synthesized for a sequential design.")
                if score > best_score:
                    best_score   = score
                    best_v_code  = v_code
                    best_err_log = ("CRITICAL: Sequential design produced 0 registers. "
                                    "Ensure all state registers use non-blocking assignments "
                                    "inside always @(posedge clk).")
                continue

            print(f"      ✅ Yosys passed: {log}")
            round_passed = True
            break

        # ── End of round bookkeeping ───────────────────────────────────────────
        if round_passed:
            print(f"\n{'═'*60}")
            print("🎉 PIPELINE COMPLETE — Structural + Functional + Physical verification passed.")
            print(f"📁 Winning design: ./workspace/{design_name}/candidate_{global_candidate_id}/")
            print(f"{'═'*60}\n")
            return True

        # Update consecutive_sim_fails counter
        # Only increment if we're stuck at score=1 (lint passes but sim fails)
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
