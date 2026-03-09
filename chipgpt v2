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

# 1. Global Client (Lazy Initialization)
openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = OpenAI()
    return openai_client

def code_hash(code: str):
    normalized = re.sub(r'\s+', ' ', code.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def truncate_log(log: str, max_lines: int = 15):
    lines = log.strip().split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + "\n... [ERRORS TRUNCATED]"
    return log

# PEER UPGRADE: Contextual +/- 1 line error extraction
def extract_primary_error(log: str):
    lines = log.strip().split('\n')
    for i, line in enumerate(lines):
        if "error" in line.lower() or "mismatch" in line.lower() or "fail" in line.lower() or "syntax" in line.lower():
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            return "\n".join(lines[start:end])
    return truncate_log(log, 5)

def parse_llm_json(raw_text: str):
    try:
        triple_tick = "`" * 3
        clean_text = re.sub(rf'{triple_tick}(?:json)?|{triple_tick}$', '', raw_text.strip(), flags=re.MULTILINE).strip()
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            return json.loads(match.group(), strict=False)
        return json.loads(clean_text, strict=False)
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON Parse Error: {e}")
        return {}

def clean_code_string(raw_code: str):
    if not raw_code:
        return ""
    clean = raw_code.strip()
    triple_tick = "`" * 3
    clean = clean.replace(triple_tick + "verilog", "")
    clean = clean.replace(triple_tick + "systemverilog", "")
    clean = clean.replace(triple_tick + "v", "")
    clean = clean.replace(triple_tick, "")
    return clean.strip()

def run_llm(prompt: str, system_prompt: str, provider: str, max_api_retries=3, cand_idx=0, force_fast_model=False):
    temps = [0.2, 0.5, 0.8] 
    efforts = ["low", "medium", "high"]
    
    for attempt in range(max_api_retries):
        try:
            if provider == "openai":
                client = get_openai_client()
                
                # RESTORED GPT-5-MINI
                model_name = "gpt-4o-mini" if force_fast_model else "gpt-5-mini" 
                
                kwargs = {
                    "model": model_name,
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "timeout": 150 # Generous timeout for GPT-5 reasoning
                }
                
                if "gpt-5" in model_name or "o1" in model_name or "o3" in model_name:
                    kwargs["reasoning_effort"] = efforts[cand_idx % 3]
                else:
                    kwargs["temperature"] = temps[cand_idx % 3]
                    
                print(f"      📡 [API] Thread {cand_idx+1}: Calling {model_name} (Attempt {attempt+1})...")
                response = client.chat.completions.create(**kwargs)
                return parse_llm_json(response.choices[0].message.content)
                
            elif provider == "groq":
                client = Groq()
                print(f"      📡 [API] Thread {cand_idx+1}: Calling Groq...")
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=temps[cand_idx % 3],
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    timeout=90
                )
                return parse_llm_json(response.choices[0].message.content)
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str:
                print(f"   ⏳ Thread {cand_idx+1} Rate Limit. Sleeping 35s...")
                time.sleep(35)
            elif "timeout" in error_str:
                print(f"   ⚠️ Thread {cand_idx+1} API Timeout! Retrying...")
            else:
                print(f"   ⚠️ Thread {cand_idx+1} Fatal API Error: {e}")
                time.sleep(5)
    return {}

def generate_specification(prompt: str, provider: str):
    print(f"   📝 Routing to CLARIFIER AGENT ({provider.upper()})...")
    system_prompt = """
    You are a Lead Hardware Specifications Engineer.
    Translate the user's ambiguous request into a strict, disambiguated technical specification.
    Return pure JSON: {"formalized_request": "...", "data_widths": "...", "latency_expectation": "...", "reset_behavior": "...", "overflow_behavior": "..."}
    """
    return run_llm(prompt, system_prompt, provider, cand_idx=0, force_fast_model=True)

def generate_architecture(spec: dict, provider: str, previous_error: str = None):
    print(f"   📐 Routing to ARCHITECT AGENT ({provider.upper()})...")
    error_context = f"\nCRITICAL FEEDBACK FROM PREVIOUS RUN:\n{previous_error}\n" if previous_error else ""
    system_prompt = f"""
    You are an elite SoC Architect. Design a strictly synthesizable hardware architecture based on this spec:
    {json.dumps(spec, indent=2)}
    {error_context}
    
    CRITICAL: Decide if this is primarily an FSM or a DATAPATH.
    You MUST return pure JSON exactly matching this schema:
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
    return run_llm(json.dumps(spec), system_prompt, provider, cand_idx=0, force_fast_model=True)

def validate_architecture(arch: dict):
    if not arch or "module_class" not in arch: return False
    if not arch.get("port_interface"): return False
    
    if arch["module_class"] == "FSM":
        if not arch.get("fsm_specific", {}).get("state_encoding"): return False
        if not arch.get("fsm_specific", {}).get("state_transitions"): return False
    return True

# PEER UPGRADE: Bulletproof Port Matcher
def ports_match(v_code: str, locked_ports_json: str):
    if not locked_ports_json: return True
    locked_ports = json.loads(locked_ports_json)
    
    m = re.search(r'module\s+\w+\s*\((.*?)\);', v_code, re.S)
    if not m: return False
    port_str = m.group(1)
    
    declarations = [p.strip() for p in re.split(r',', port_str)]
    
    for p in locked_ports:
        name = str(p.get("name", ""))
        direction = str(p.get("direction", ""))
        
        found = False
        for decl in declarations:
            if re.search(rf'\b{direction}\b', decl) and re.search(rf'\b{name}\b', decl):
                found = True
                break
        if not found: return False
    return True

def review_hardware(v_code: str, provider: str):
    system_prompt = """
    You are an elite Silicon Verification Engineer. 
    Analyze this Verilog RTL strictly for:
    1. Inferred Latches
    2. Missing reset conditions
    3. Combinational loops
    
    CRITICAL: Output a patch. DO NOT REWRITE THE ENTIRE MODULE. If perfect, return "PASSED".
    Return JSON: {"status": "PASSED" or "REJECTED", "fixed_code": "[CODE]"}
    """
    return run_llm(v_code, system_prompt, provider, cand_idx=0, force_fast_model=True)

def generate_hardware(prompt: str, provider: str, design_name: str, architecture: dict, previous_ports: str = None, previous_error: str = None, best_v_code: str = None, cand_idx: int = 0):
    port_rule = "2. STRICT INTERFACE: The Verilog module's port declarations MUST perfectly match the Architect's `port_interface` array."
    if previous_ports:
        port_rule = f"2. STRICT INTERFACE CRITICAL: You MUST reuse this exact previous interface: {previous_ports}"

    error_context = ""
    if previous_error and best_v_code:
        error_context = f"""
        === EVOLUTIONARY MUTATION REQUIRED ===
        Your previous best design failed Simulation/Synthesis. ERROR: {previous_error}
        YOUR BEST PREVIOUS VERILOG: {best_v_code}
        CRITICAL: DO NOT rewrite the module from scratch. Only modify lines related to the error.
        ======================================
        """
    elif previous_error:
        error_context = f"=== SYNTAX ERROR IN PREVIOUS ATTEMPT ===\nERROR: {previous_error}\nFix the Verilog.\n========================================"

    system_prompt = f"""
    You are an elite RTL design engineer.
    {error_context}
    Implement exactly this architecture designed by the Senior Architect:
    {json.dumps(architecture, indent=2)}

    CRITICAL SILICON RULES:
    1. TOP MODULE NAME: The main module MUST be named exactly: `{design_name}`.
    {port_rule}
    3. STRICT VERILOG-2001 ONLY: No SystemVerilog (logic, always_ff). No initial blocks inside modules.
    4. CONCURRENCY: Sequential logic MUST use non-blocking (`<=`) in always @(posedge clk...). Combinational MUST use blocking (`=`) in always @(*).
    5. NO LATCHES: Every combinational always @(*) block MUST assign default values to all driven signals at the very top.
    6. TESTBENCH: TB must begin with `timescale 1ns/1ps`. 
       - MUST include a self-checking Golden Reference Model to cross-check outputs.
       - MUST include a hard timeout: `initial begin #10000; $display("FAIL: Timeout"); $finish; end`
       - AT THE VERY END, print exactly: `$display("SIMULATION_SUCCESS");` and call `$finish;`.

    Return pure JSON: {{"structural_reasoning": "...", "implemented_fsm_states": [...], "implemented_datapath_blocks": [...], "signal_table": [...], "verilog_code": "...", "testbench_code": "..."}}
    """
    
    design = run_llm(prompt, system_prompt, provider, cand_idx=cand_idx)
    v_code = clean_code_string(design.get("verilog_code") or "")
    
    if len(v_code) > 50:
        review = review_hardware(v_code, provider)
        if review.get("status") == "REJECTED" and review.get("fixed_code"):
            fixed = clean_code_string(review.get("fixed_code"))
            if 0.8 < (len(fixed) / len(v_code)) < 1.2:
                print(f"      ✅ [Reviewer] Thread {cand_idx+1}: Safely patched a bug before Verilator!")
                design["verilog_code"] = fixed
            else:
                print(f"      ⚠️ [Reviewer] Thread {cand_idx+1}: Hallucinated a massive rewrite. Discarding review.")
                
    return design

def save_to_workspace(design_name: str, design: dict, global_candidate_id: int):
    workspace_dir = f"./workspace/{design_name}/candidate_{global_candidate_id}"
    os.makedirs(f"{workspace_dir}/src", exist_ok=True)
    os.makedirs(f"{workspace_dir}/tb", exist_ok=True)

    v_code = clean_code_string(design.get("verilog_code") or "")
    tb_code = clean_code_string(design.get("testbench_code") or "")

    v_file = f"{workspace_dir}/src/{design_name}.v"
    tb_file = f"{workspace_dir}/tb/{design_name}_tb.v"
    
    with open(v_file, "w") as f: f.write(v_code)
    with open(tb_file, "w") as f: f.write(tb_code)
    return v_file, tb_file, v_code

def verify_with_verilator(v_file: str, design_name: str, v_code: str):
    if not re.search(r'\bmodule\s+' + re.escape(design_name) + r'\b', v_code):
        return False, f"%Error: Top module '{design_name}' is missing or named incorrectly."

    cmd = ["verilator", "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Werror-WIDTH", v_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0: return True, "Clean"
        return False, truncate_log(result.stderr)
    except subprocess.TimeoutExpired:
        return False, "Verilator Timeout!"

def verify_with_iverilog(v_file: str, tb_file: str, design_name: str, global_candidate_id: int):
    workspace_dir = f"./workspace/{design_name}/candidate_{global_candidate_id}"
    output_bin = f"{workspace_dir}/{design_name}_sim.vvp"
    
    compile_cmd = ["iverilog", "-o", output_bin, v_file, tb_file]
    compile_result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if compile_result.returncode != 0:
        return False, f"Compilation Failed:\n{truncate_log(compile_result.stderr)}"
        
    sim_cmd = ["vvp", output_bin]
    try:
        sim_result = subprocess.run(sim_cmd, capture_output=True, text=True, timeout=20)
        if sim_result.returncode != 0:
            return False, f"Runtime Failed:\n{truncate_log(sim_result.stderr)}"
        if any(kw in sim_result.stdout for kw in ["ERROR", "Error", "FAIL", "Fail"]):
            return False, f"Verification Failed:\n{truncate_log(sim_result.stdout, 15)}"
        
        if "SIMULATION_SUCCESS" not in sim_result.stdout or "finish" not in sim_result.stdout.lower():
            return False, "CRITICAL: Simulation exited prematurely or timeout hit. Missing SIMULATION_SUCCESS."
            
        return True, "Simulation Clean"
    except subprocess.TimeoutExpired:
        return False, "Simulation Timeout (Infinite Loop in testbench)!"

def verify_with_yosys(v_file: str):
    cmd = ["yosys", "-p", f"read_verilog {v_file}; proc; opt; fsm; memory; check -assert; prep; stat"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        combined_log = result.stderr + "\n" + result.stdout
        if result.returncode == 0 and "ERROR" not in combined_log:
            dffs = len(re.findall(r'\$(?:a?dff|dffe|sdff)', combined_log, re.IGNORECASE))
            cells = re.search(r'Number of cells:\s+(\d+)', combined_log)
            cell_count = cells.group(1) if cells else '0'
            stats_msg = f"Cells: {cell_count}, Registers (DFFs): {dffs}"
            return True, stats_msg
        return False, truncate_log(combined_log, 20)
    except subprocess.TimeoutExpired:
        return False, "Yosys Timeout!"

def autonomous_build_loop(base_prompt: str, design_name: str, provider: str, max_retries: int = 10, candidates_per_round: int = 3):
    print(f"🚀 Starting Evolutionary Build: '{base_prompt}' using {provider.upper()}")
    clarified_spec = generate_specification(base_prompt, provider)
    
    arch_hashes = set()
    code_hashes = set()
    locked_ports = None 
    error_log = None
    architecture = None
    architecture_flawed = True 
    
    best_score = -1 
    best_v_code = None
    best_err_log = None
    global_candidate_id = 0
    consecutive_sim_failures = 0
    
    for attempt in range(max_retries):
        print(f"\n🔄 Generation {attempt + 1}/{max_retries}...")
        
        if consecutive_sim_failures >= 3:
            print("   🔄 Forcing Architect to redesign blueprint (Stuck on logic failures)...")
            architecture_flawed = True
            consecutive_sim_failures = 0
            best_score = -1 
            best_v_code = None

        if architecture_flawed:
            architecture = generate_architecture(clarified_spec, provider, error_log)
            arch_hash = code_hash(json.dumps(architecture, sort_keys=True))
            
            if not validate_architecture(architecture):
                print("   ⚠️ Architect returned invalid blueprint.")
                error_log = "CRITICAL: Architecture is missing required states or definitions."
                continue
            
            if arch_hash in arch_hashes:
                print("   ⚠️ Architect generated duplicate blueprint. Forcing variation.")
                error_log = "CRITICAL: Do not output the exact same blueprint. Change your structural approach."
                continue
                
            arch_hashes.add(arch_hash)
            architecture_flawed = False 
        else:
            print("   📐 Architect Agent: Reusing stable architectural blueprint.")
        
        if best_v_code:
            print(f"   🧬 Evolution Engine: Mutating previous best candidate (Score: {best_score}/2)...")
        else:
            print(f"   🛠️ Generator Agent: Spawning {candidates_per_round} parallel RTL candidates (Async Mode)...")
            
        candidates = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=candidates_per_round) as executor:
            futures = [executor.submit(generate_hardware, base_prompt, provider, design_name, architecture, locked_ports, best_err_log or error_log, best_v_code, i) for i in range(candidates_per_round)]
            
            try:
                for future in concurrent.futures.as_completed(futures, timeout=180):
                    design = future.result()
                    v_code = clean_code_string(design.get("verilog_code") or "")
                    
                    if len(v_code) > 50:
                        if locked_ports and not ports_match(v_code, locked_ports):
                            print(f"   ⚠️ Candidate hallucinated/mutated the port interface. Discarding.")
                            continue

                        current_v_hash = code_hash(v_code)
                        if current_v_hash not in code_hashes:
                            code_hashes.add(current_v_hash)
                            
                            if architecture.get("module_class") == "DATAPATH" and not re.search(r'reg\s*\[[^\]]+\]\s*\w+\s*\[[^\]]+\]', v_code):
                                print(f"   ⚠️ Candidate hallucinated a datapath without a memory array. Discarding.")
                                continue
                                
                            candidates.append(design)
            except concurrent.futures.TimeoutError:
                print("   ⚠️ ThreadPoolExecutor Timeout! The API hung indefinitely. Restarting round.")
        
        if not candidates:
            print("   ❌ All generated candidates were empty, rejected, or duplicates. Retrying...")
            error_log = "CRITICAL: You returned empty Verilog code, mutated locked ports, or exact duplicates."
            continue
            
        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture.get("port_interface"))

        round_passed = False
        round_sim_passed = False
        
        for idx, design in enumerate(candidates):
            global_candidate_id += 1
            print(f"   🔬 Testing Candidate {global_candidate_id}...")
            
            v_file, tb_file, v_code = save_to_workspace(design_name, design, global_candidate_id)
            score = 0
            
            passed_lint, lint_log = verify_with_verilator(v_file, design_name, v_code)
            if not passed_lint:
                err_msg = extract_primary_error(lint_log)
                print(f"      ❌ Verilator Failed: {err_msg}")
                if score >= best_score:
                    best_score = score
                    best_err_log = f"Syntax Error:\n{err_msg}"
                continue
            
            score = 1
            print(f"      ✅ Verilator Passed.")
            
            passed_sim, sim_log = verify_with_iverilog(v_file, tb_file, design_name, global_candidate_id)
            if not passed_sim:
                err_msg = extract_primary_error(sim_log)
                print(f"      ❌ Icarus Verilog Failed: {err_msg}")
                if score >= best_score:
                    best_score = score
                    best_v_code = v_code 
                    best_err_log = f"Simulation Error:\n{err_msg}"
                continue
                
            score = 2
            round_sim_passed = True
            print(f"      ✅ Icarus Verilog Passed.")
            
            passed_synth, synth_log = verify_with_yosys(v_file)
            if not passed_synth:
                err_msg = extract_primary_error(synth_log)
                print(f"      ❌ Yosys Failed: {err_msg}")
                if score >= best_score:
                    best_score = score
                    best_v_code = v_code
                    best_err_log = f"Synthesis Issue: {err_msg}"
                continue
            
            if re.search(r"Registers \(DFFs\):\s*0", synth_log, re.IGNORECASE) and ("clk" in base_prompt or "clock" in base_prompt):
                print(f"      ⚠️ Yosys Warning: 0 registers synthesized.")
                if score >= best_score:
                    best_score = score
                    best_v_code = v_code
                    best_err_log = "CRITICAL: Sequential logic expected but 0 registers synthesized."
                continue
                
            print(f"      ✅ Yosys Passed: {synth_log}")
            round_passed = True
            break 

        if round_passed:
            print("\n🎉 Full Pipeline Passed! Structural, Functional, and Physical Verification Complete.")
            print(f"📁 Winning design saved in ./workspace/{design_name}/candidate_{global_candidate_id}/")
            return True
        else:
            if best_score == 1 and not round_sim_passed:
                consecutive_sim_failures += 1
            else:
                consecutive_sim_failures = 0
                
            print(f"   🤖 End of Generation {attempt + 1}. Highest Score: {best_score}/2. (Sim Fails: {consecutive_sim_failures})")
            
    print("\n🚨 Max retries reached.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChipGPT: Autonomous RTL Generator")
    parser.add_argument("prompt", type=str, help="What do you want to build?")
    parser.add_argument("--name", type=str, default="my_module", help="Name of the top module")
    parser.add_argument("--provider", type=str, choices=["openai", "groq", "gemini"], default="openai")
    args = parser.parse_args()
    autonomous_build_loop(args.prompt, args.name, args.provider)
