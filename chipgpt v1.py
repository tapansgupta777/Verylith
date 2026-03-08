import os
import json
import argparse
import subprocess
import hashlib
import re
import time

from openai import OpenAI
from groq import Groq
import google.generativeai as genai

def code_hash(code: str):
    # Normalize whitespace to catch semantic duplicates hiding in formatting
    normalized = re.sub(r'\s+', ' ', code.strip())
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()

def truncate_log(log: str, max_lines: int = 15):
    lines = log.strip().split('\n')
    if len(lines) > max_lines:
        return '\n'.join(lines[:max_lines]) + "\n... [ERRORS TRUNCATED: Focus on fixing the first errors above]"
    return log

def parse_llm_json(raw_text: str):
    try:
        # Find the first opening brace and the last closing brace
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}')
        
        # If both exist and are in the correct order, slice out the pure JSON
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = raw_text[start_idx:end_idx+1]
            return json.loads(json_str, strict=False)
        
        # Fallback if no brackets are found
        return json.loads(raw_text, strict=False)
        
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON Parse Error: {e}")
        return {}

def run_llm(prompt: str, system_prompt: str, provider: str, max_api_retries=3):
    """Helper function to cleanly route API calls with Rate Limit protection"""
    for attempt in range(max_api_retries):
        try:
            if provider == "openai":
                client = OpenAI()
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return parse_llm_json(response.choices[0].message.content)

            elif provider == "groq":
                client = Groq()
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                return parse_llm_json(response.choices[0].message.content)

            elif provider == "gemini":
                if not hasattr(run_llm, "gemini_init"):
                    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
                    run_llm.gemini_init = True
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(
                    system_prompt + "\n\nUser Request: " + prompt,
                    generation_config={"temperature": 0.1}
                )
                return parse_llm_json(response.text)
                
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str or "exhausted" in error_str:
                print(f"   ⏳ API Rate Limit hit. Sleeping for 35 seconds before retry ({attempt + 1}/{max_api_retries})...")
                time.sleep(35)
            else:
                print(f"   ⚠️ Fatal API Error: {e}")
                return {}
                
    print("   🚨 Max API rate-limit retries reached. Returning empty JSON.")
    return {}

def generate_specification(prompt: str, provider: str):
    """AGENT 0: The Clarifier. Removes human ambiguity from the prompt."""
    print(f"   📝 Routing to CLARIFIER AGENT ({provider.upper()}) to formalize the spec...")
    
    system_prompt = """
    You are a Lead Hardware Specifications Engineer.
    Translate the user's ambiguous request into a strict, disambiguated technical specification.
    Define missing parameters logically (e.g., synchronous vs asynchronous reset, latencies, edge cases).
    
    Return pure JSON:
    {
        "formalized_request": "The exact goal...",
        "data_widths": "e.g., 8-bit, 32-bit",
        "latency_expectation": "e.g., 1-cycle, combinational",
        "reset_behavior": "e.g., synchronous active-high",
        "overflow_behavior": "e.g., wraparound, saturate"
    }
    """
    return run_llm(prompt, system_prompt, provider)

def generate_architecture(spec: dict, provider: str, previous_error: str = None):
    """AGENT 1: The Architect. Decides between FSM or Datapath schemas."""
    print(f"   📐 Routing to ARCHITECT AGENT ({provider.upper()}) to design the blueprint...")
    
    error_context = f"\nCRITICAL CONTEXT FROM PREVIOUS PIPELINE FAILURE:\n{previous_error}\n" if previous_error else ""
    
    system_prompt = f"""
    You are an elite System-on-Chip (SoC) Architect.
    Design a strictly synthesizable hardware architecture based on this clarified specification:
    {json.dumps(spec, indent=2)}
    
    Do NOT write any Verilog code. Only output the architectural specification.
    {error_context}

    CRITICAL: Decide if this design is primarily an FSM (Control logic) or a DATAPATH (Memory, Arithmetic, FIFO).
    
    If it is an FSM, return pure JSON exactly matching this schema:
    {{
        "module_class": "FSM",
        "port_interface": [{{"name": "clk", "direction": "input", "width": 1}}],
        "state_encoding": {{"IDLE": 0, "EXECUTE": 1}},
        "registers": [{{"name": "counter", "width": 8}}],
        "datapath_signals": [{{"name": "sum", "width": 8}}],
        "state_transitions": [{{"from": "IDLE", "to": "EXECUTE", "condition": "start == 1"}}],
        "state_outputs": {{"IDLE": {{"load_en": 0}}, "EXECUTE": {{"load_en": 1}}}}
    }}

    If it is a DATAPATH, return pure JSON exactly matching this schema:
    {{
        "module_class": "DATAPATH",
        "port_interface": [{{"name": "clk", "direction": "input", "width": 1}}],
        "memory_blocks": [{{"name": "fifo_ram", "width": 8, "depth": 16}}],
        "registers": [{{"name": "wr_ptr", "width": 4}}],
        "datapath_signals": [{{"name": "full", "width": 1}}]
    }}
    """
    return run_llm(json.dumps(spec), system_prompt, provider)

def generate_hardware(prompt: str, provider: str, design_name: str, architecture: dict, previous_ports: str = None, previous_error: str = None):
    """AGENT 2: The Generator. Adapts to FSM or Datapath with strict latch prevention and reasoning."""
    print(f"   🛠️ Routing to GENERATOR AGENT ({provider.upper()}) to write RTL...")
    
    port_rule = "2. STRICT INTERFACE: The Verilog module's port declarations MUST perfectly match the Architect's `port_interface` array."
    if previous_ports:
        port_rule = f"2. STRICT INTERFACE CRITICAL: You MUST reuse this exact previous interface: {previous_ports}"

    error_context = f"\n=== CRITICAL FEEDBACK FROM PREVIOUS FAILURE ===\n{previous_error}\nDO NOT REPEAT THIS MISTAKE.\n===============================================\n" if previous_error else ""

    system_prompt = f"""
    You are an elite RTL design engineer. Your goal is to write perfectly synthesizable Verilog-2001.
    {error_context}
    Implement exactly this architecture designed by the Senior Architect:
    {json.dumps(architecture, indent=2)}

    CRITICAL SILICON RULES:
    1. TOP MODULE NAME: The main module MUST be named exactly: `{design_name}`.
    {port_rule}
    3. STRICT VERILOG-2001 ONLY: You are targeting a strict compiler. The following are STRICTLY FORBIDDEN:
       - `logic` (use `reg` or `wire`)
       - `always_ff` / `always_comb` (use `always @(posedge clk)` / `always @(*)`)
       - `$urandom` / `$urandom_range` (use `$random`)
       - `int` (use `integer`)
       - NEVER declare variables inside procedural blocks (like inside an `initial`, `always`, or `for` loop). 
         [BAD]: initial begin reg temp; temp = 1; end
         [GOOD]: reg temp; initial begin temp = 1; end
    4. CONCURRENCY: Sequential logic MUST use non-blocking (`<=`) in an `always @(posedge clk...)` block. Combinational logic MUST use blocking (`=`) in a separate `always @(*)` block. No exceptions.
    5. ARCHITECTURE COMPLIANCE: 
       - If `module_class` is "FSM", implement a deterministic state machine using `case(state)`.
       - If `module_class` is "DATAPATH", cleanly implement the memory arrays, pointers, and flags without a state machine.
    6. NO LATCHES: Every combinational `always @(*)` block MUST assign default values to all its driven signals at the very top.
    7. RESETS: All sequential registers MUST be explicitly initialized in the reset condition.
    8. TESTBENCH: TB must begin with `` `timescale 1ns/1ps `` and include: `initial begin $dumpfile("{design_name}.vcd"); $dumpvars(0, {design_name}_tb); end`. Run randomized checks and print "ERROR" on mismatch.
    9. SILENT TRUNCATION PREVENTION: Calculate register widths mathematically. To store a maximum value of N (like a depth or counter), the register MUST be at least ceil(log2(N+1)) bits wide. Never compare a register to a value larger than its maximum representable unsigned number.
    10. INVARIANT ASSERTIONS: To prevent spec drift, the testbench MUST include runtime assertions using standard Verilog. Check for impossible conditions (e.g., `if (wr_en && full) $display("ERROR: Write when full!");`).
    
    Return pure JSON:
    {{
     "structural_reasoning": "Plan your state register widths, datapath widths, reset behavior, and concurrency isolation here BEFORE writing code...",
     "implemented_fsm_states": ["IDLE"],
     "implemented_datapath_blocks": ["fifo_mem", "wr_ptr"],
     "signal_table": [{{"name": "wr_ptr", "type": "reg", "width": 4}}],
     "verilog_code": "// module code",
     "testbench_code": "// testbench code"
    }}
    """
    return run_llm(prompt, system_prompt, provider)

def run_critic_agent(design: dict, architecture: dict, base_prompt: str, provider: str, previous_error: str = None):
    """AGENT 3: The Critic. Audits for architecture compliance and synthesis hazards."""
    print(f"   🕵️‍♂️ Routing to CRITIC AGENT ({provider.upper()}) for logic review...")
    
    error_context = f"\nCRITICAL CONTEXT FROM PREVIOUS PIPELINE FAILURE:\n{previous_error}\n" if previous_error else ""
    
    user_prompt = f"""
    Junior engineer task: '{base_prompt}'
    {error_context}
    
    Architect Blueprint: {json.dumps(architecture, indent=2)}
    Implemented States: {json.dumps(design.get('implemented_fsm_states', []))}
    Implemented Datapath: {json.dumps(design.get('implemented_datapath_blocks', []))}
    Signal Table: {json.dumps(design.get('signal_table', []))}
    
    Verilog: {design.get('verilog_code')}
    Testbench: {design.get('testbench_code')}
    
    Review for critical flaws:
    1. ARCHITECTURE CONTRACT: Does the RTL strictly follow the Architect's module_class?
    2. CONCURRENCY & RESETS: Are non-blocking (`<=`) used strictly for sequential logic? Are there ANY blocking (`=`) assignments leaking into clocked blocks? Are ALL sequential registers properly initialized via reset logic?
    3. SYNTHESIS HAZARDS: Check for width mismatches (e.g., FSM parameter width vs state register width), accidental latches, uninitialized registers, and combinational loops.
    4. TESTBENCH: Does it run exhaustive tests and check edge cases?
    
    Return pure JSON:
    {{
        "pass": true,
        "architecture_flawed": false,
        "feedback": "Detailed explanation..."
    }}
    """
    
    system_instruction = "You are a Senior Staff RTL Verification Engineer. Review the submitted code against the architectural blueprint."
    return run_llm(user_prompt, system_instruction, provider)

def save_to_workspace(design_name: str, design: dict):
    workspace_dir = f"./workspace/{design_name}"
    src_dir = f"{workspace_dir}/src"
    tb_dir = f"{workspace_dir}/tb"
    
    os.makedirs(src_dir, exist_ok=True, mode=0o777)
    os.makedirs(tb_dir, exist_ok=True, mode=0o777)

    v_file = f"{src_dir}/{design_name}.v"
    tb_file = f"{tb_dir}/{design_name}_tb.v"

    v_code = design.get("verilog_code") or ""
    tb_code = design.get("testbench_code") or ""

    with open(v_file, "w") as f:
        f.write(v_code)
    with open(tb_file, "w") as f:
        f.write(tb_code)

    return v_file, tb_file, v_code

def verify_with_verilator(v_file: str):
    print("   🛠️ Stage 1: Running Verilator structural check...")
    # NEW: Removed -Wno-fatal and added -Werror-WIDTH to instantly catch truncation
    cmd = ["verilator", "--lint-only", "-Wall", "-Wno-DECLFILENAME", "-Werror-WIDTH", v_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            return True, "Clean"
        return False, truncate_log(result.stderr)
    except subprocess.TimeoutExpired:
        return False, "Verilator Check Timed Out!"

def verify_with_iverilog(v_file: str, tb_file: str, design_name: str):
    print("   🧪 Stage 2: Running Icarus Verilog functional simulation...")
    workspace_dir = f"./workspace/{design_name}"
    output_bin = f"{workspace_dir}/{design_name}_sim.vvp"
    
    compile_cmd = ["iverilog", "-o", output_bin, v_file, tb_file]
    try:
        compile_result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=15)
        if compile_result.returncode != 0:
            return False, f"Compilation Failed:\n{truncate_log(compile_result.stderr)}"
    except subprocess.TimeoutExpired:
        return False, "Icarus Compilation Timed Out!"
        
    sim_cmd = ["vvp", f"{design_name}_sim.vvp"]
    try:
        sim_result = subprocess.run(sim_cmd, cwd=workspace_dir, capture_output=True, text=True, timeout=20)
        
        if sim_result.returncode != 0:
            return False, f"Runtime Failed:\n{truncate_log(sim_result.stderr)}"
            
        error_keywords = ["ERROR", "Error", "FAIL", "Fail", "Mismatch", "mismatch"]
        if any(keyword in sim_result.stdout for keyword in error_keywords):
            return False, f"Logic Verification Failed:\n{truncate_log(sim_result.stdout, 15)}"
            
        return True, "Simulation Clean"
    except subprocess.TimeoutExpired:
        return False, "CRITICAL ERROR: Simulation timed out. Check for infinite loops in testbench."

def verify_with_yosys(v_file: str):
    print("   ⚙️ Stage 3: Running Yosys physical synthesis check...")
    cmd = ["yosys", "-p", f"read_verilog {v_file}; proc; opt; check; prep; synth -run begin:check; stat"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        combined_log = result.stderr + "\n" + result.stdout
        
        if result.returncode == 0 and "ERROR" not in combined_log:
            dff_matches = re.findall(r'(\d+)\s+.*DFF.*', combined_log, re.IGNORECASE)
            total_dffs = sum(int(m) for m in dff_matches)
            cells = re.search(r'Number of cells:\s+(\d+)', combined_log)
            cell_count = cells.group(1) if cells else '0'
            
            stats_msg = f"Cells: {cell_count}, Registers (DFFs): {total_dffs}"
            print(f"      📊 Yosys Stats -> {stats_msg}")
            return True, stats_msg
        
        return False, truncate_log(combined_log, 20)
    except subprocess.TimeoutExpired:
        return False, "Yosys Synthesis Timed Out!"

def autonomous_build_loop(base_prompt: str, design_name: str, provider: str, max_retries: int = 10):
    print(f"🚀 Starting Quad-Agent Build: '{base_prompt}' using {provider.upper()}")
    
    # AGENT 0: Disambiguate the human prompt
    clarified_spec = generate_specification(base_prompt, provider)
    
    submitted_hashes = set()
    locked_ports = None 
    error_log = None
    architecture = None
    architecture_flawed = True 
    
    for attempt in range(max_retries):
        print(f"\n🔄 Attempt {attempt + 1}/{max_retries}...")
        
        if architecture_flawed:
            architecture = generate_architecture(clarified_spec, provider, error_log)
            if not architecture or "module_class" not in architecture:
                print("   ⚠️ Architect returned invalid blueprint. Forcing redesign.")
                error_log = "CRITICAL: You failed to output a valid JSON architecture blueprint."
                continue # Skip the rest and retry the Architect immediately
            architecture_flawed = False 
        else:
            print("   📐 Architect Agent: Reusing stable architectural blueprint.")
        
        # WE ARE FINALLY PASSING THE ERROR LOG TO THE GENERATOR!
        design = generate_hardware(base_prompt, provider, design_name, architecture, locked_ports, error_log)
        v_code = design.get("verilog_code") or ""
        
        if not locked_ports and architecture.get("port_interface"):
            locked_ports = json.dumps(architecture.get("port_interface"))
        
        current_v_hash = code_hash(v_code)
        is_duplicate = current_v_hash in submitted_hashes
        submitted_hashes.add(current_v_hash)
        
        passed = False
        if is_duplicate:
            print("   ⚠️ AI resubmitted identical code! Bypassing checks to save time.")
            error_log = "CRITICAL ERROR: You submitted identical Verilog code. Your RTL logic must change drastically."
        else:
            v_file, tb_file, _ = save_to_workspace(design_name, design)
            
            passed_lint, lint_log = verify_with_verilator(v_file)
            if not passed_lint:
                print("   ❌ Verilator Failed (Syntax Error). Skipping Critic review.")
                error_log = lint_log
            else:
                print("   ✅ Verilator Passed.")
                
                critic_review = run_critic_agent(design, architecture, base_prompt, provider, error_log)
                if not critic_review.get("pass", False):
                    print("   🛑 Critic Agent Rejected the Code! (Logical Flaw Found)")
                    error_log = f"Senior engineer review detected the following flaw:\n{critic_review.get('feedback')}\n\nFix the RTL to match the spec. Do NOT weaken or delete tests from the testbench to force a pass."
                    architecture_flawed = critic_review.get("architecture_flawed", False)
                else:
                    print("   ✅ Critic Agent Approved the Logic.")
                    
                    passed_sim, sim_log = verify_with_iverilog(v_file, tb_file, design_name)
                    if not passed_sim:
                        print("   ❌ Icarus Verilog Failed.")
                        error_log = sim_log
                    else:
                        print("   ✅ Icarus Verilog Passed.")
                        
                        passed_synth, synth_log_or_stats = verify_with_yosys(v_file)
                        if not passed_synth:
                            print("   ❌ Yosys Synthesis Failed.")
                            error_log = synth_log_or_stats
                        else:
                            print("   ✅ Yosys Synthesis Passed.")
                            
                            if "Registers (DFFs): 0" in synth_log_or_stats and ("clk" in base_prompt or "clock" in base_prompt):
                                print("   ⚠️ Yosys Warning: Design contains a clock but synthesized 0 registers. Rejecting.")
                                passed = False
                                error_log = f"Synthesis Statistics:\n{synth_log_or_stats}\n\nCRITICAL ERROR: Sequential logic is expected but 0 registers synthesized. Fix your clocks."
                            else:
                                passed = True

        if passed:
            print("\n🎉 Full Pipeline Passed! Structural, Functional, and Physical Verification Complete.")
            print(f"📁 Final design saved in ./workspace/{design_name}")
            print(f"🌊 Waveform file (.vcd) generated in ./workspace/{design_name} for viewing in GTKWave.")
            return True
        else:
            print("   🤖 Feeding context and error logs back into the Quad-Agent system...")
            
    print("\n🚨 Max retries reached.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChipGPT: Autonomous RTL Generator")
    parser.add_argument("prompt", type=str, help="What do you want to build?")
    parser.add_argument("--name", type=str, default="my_module", help="Name of the top module")
    parser.add_argument("--provider", type=str, choices=["openai", "groq", "gemini"], default="openai", 
                        help="Choose which LLM API to use (default: openai)")
    
    args = parser.parse_args()
    autonomous_build_loop(args.prompt, args.name, args.provider)
