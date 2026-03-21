"""
skeletons.py — ChipGPT RTL Skeleton Library
============================================
Each skeleton encodes the structural invariants for a design class.
Skeletons are NOT complete implementations — they are starting points
that encode patterns which must never change (reset structure, handshake
logic, non-blocking discipline) while leaving parameters and behavioral
branches for the LLM to fill.

SKELETON SCOPE POLICY:
- Skeletons match the *most common* form of each design class.
- If the prompt describes a variant not covered (e.g. gray counter,
  bidirectional shift with parallel load), the skeleton is still used
  as a structural reference but the LLM has more latitude.
- Frozen lines (reset structure, assign logic, sensitivity lists) are
  listed in each entry's `invariants` field for post-generation checking.
- The module name placeholder is literally: <MODULE_NAME>
- Width placeholder: <WIDTH>  (default 8)
"""

from typing import Dict, List

# Each entry:
#   skeleton  : str   — Verilog template, <MODULE_NAME> and <WIDTH> are replaced
#   invariants: list  — regex patterns that MUST appear in generated RTL
#   keywords  : list  — prompt keywords that trigger this skeleton (priority order)
#   notes     : str   — designer notes on scope/limitations

SKELETON_LIBRARY: Dict[str, dict] = {

    # ------------------------------------------------------------------
    "counter": {
        "keywords": [
            "up/down counter", "up down counter", "bidirectional counter",
            "loadable counter", "saturating counter", "binary counter",
            "counter",
        ],
        "notes": (
            "Covers synchronous binary counter with optional load, enable, "
            "overflow/underflow flags. Does NOT cover gray, LFSR, or ring counters."
        ),
        "invariants": [
            r"always\s*@\s*\(\s*posedge\s+clk\s+or\s+negedge\s+rst_n\s*\)",
            r"if\s*\(\s*!rst_n\s*\)",
            r"<=\s*[0'\{]",   # reset to 0, 1'b0, or {WIDTH{...}}
        ],
        "skeleton": """\
// ChipGPT Skeleton: synchronous binary counter with async active-low reset
// LLM: fill in WIDTH, signal names, enable/load/overflow logic as specified.
// DO NOT change: sensitivity list, reset structure, non-blocking discipline.
module <MODULE_NAME> #(parameter WIDTH = <WIDTH>) (
    input                  clk,
    input                  rst_n,      // async active-low reset
    input                  enable,
    // LLM: add load_en, load_data, up_down ports here if spec requires them
    output reg [WIDTH-1:0] count_out
    // LLM: add overflow, underflow output wires here if spec requires them
);
    // LLM: declare combinational flag wires here (assign statements below)

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count_out <= {WIDTH{1'b0}};
            // LLM: reset any other regs here
        end else begin
            if (enable) begin
                count_out <= count_out + 1;
                // LLM: replace above with load/up-down logic as required by spec
            end
        end
    end

    // LLM: add combinational assign statements for overflow/underflow flags here
    // Example: assign overflow = (count_out == {WIDTH{1'b1}}) && enable;
endmodule""",
    },

    # ------------------------------------------------------------------
    "shift_register": {
        "keywords": [
            "serial-in parallel-out", "sipo", "parallel-in serial-out", "piso",
            "serial-in serial-out", "siso", "shift register", "shift reg",
        ],
        "notes": (
            "Covers SIPO (left or right shift). Direction is spec-dependent — "
            "LLM fills the concatenation. Does NOT cover barrel shifters or "
            "multi-tap feedback shift registers."
        ),
        "invariants": [
            r"always\s*@\s*\(\s*posedge\s+clk\s+or\s+negedge\s+rst_n\s*\)",
            r"if\s*\(\s*!rst_n\s*\)",
            r"<=\s*[0'\{]",
        ],
        "skeleton": """\
// ChipGPT Skeleton: serial-in parallel-out shift register, async active-low reset
// LLM: fill WIDTH, shift direction (left=LSB-in, right=MSB-in), shift_en condition.
// DO NOT change: sensitivity list, reset structure, non-blocking discipline.
module <MODULE_NAME> #(parameter WIDTH = <WIDTH>) (
    input                  clk,
    input                  rst_n,       // async active-low reset
    input                  shift_en,
    input                  serial_in,
    output reg [WIDTH-1:0] parallel_out
    // LLM: add direction control port if spec requires it
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            parallel_out <= {WIDTH{1'b0}};
        end else if (shift_en) begin
            // LLM: replace with correct direction:
            // LEFT shift (LSB-in):  parallel_out <= {parallel_out[WIDTH-2:0], serial_in};
            // RIGHT shift (MSB-in): parallel_out <= {serial_in, parallel_out[WIDTH-1:1]};
            parallel_out <= {parallel_out[WIDTH-2:0], serial_in};  // DEFAULT: left shift
        end
    end
endmodule""",
    },

    # ------------------------------------------------------------------
    "skid_buffer": {
        "keywords": [
            "skid buffer", "skid_buffer", "ready/valid pipeline",
            "pipeline register", "ready valid pipeline", "pipeline stage",
            "backpressure buffer",
        ],
        "notes": (
            "Canonical 1-word skid buffer. The s_ready equation and the drain/push "
            "priority are FROZEN — do not alter them. Width and port names are fillable. "
            "This is the only correct implementation for a 1-entry ready/valid buffer."
        ),
        "invariants": [
            # s_ready assign must reference m_ready AND a negated buffer term
            r"__fn:s_ready_uses_buf_valid_and_m_ready",
            # Async reset sensitivity
            r"always\s*@\s*\(\s*posedge\s+clk\s+or\s+negedge\s+rst_n\s*\)",
            # Active-low reset check
            r"if\s*\(\s*!rst_n\s*\)",
            # Some register must be reset to 0 in the reset block (any name)
            r"<=\s*[01'\{]",
        ],
        "skeleton": """\
// ChipGPT Skeleton: canonical 1-word skid buffer (ready/valid handshake)
// FROZEN: s_ready equation, drain-before-push priority, reset values.
// LLM: fill WIDTH only. Do NOT alter the handshake or priority logic.
module <MODULE_NAME> #(parameter WIDTH = <WIDTH>) (
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

    // FROZEN: this equation is the defining invariant of a skid buffer
    assign s_ready = !buf_valid || m_ready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            buf_valid <= 1'b0;
            m_valid   <= 1'b0;
            m_data    <= {WIDTH{1'b0}};
            buf_data  <= {WIDTH{1'b0}};
        end else begin
            // DRAIN path: master consumed output, or output slot is empty
            if (m_ready || !m_valid) begin
                if (buf_valid) begin
                    // Drain skid slot to output register
                    m_valid   <= 1'b1;
                    m_data    <= buf_data;
                    buf_valid <= 1'b0;
                end else if (s_valid) begin
                    // Pass source directly through (no buffering needed)
                    m_valid   <= 1'b1;
                    m_data    <= s_data;
                end else begin
                    m_valid   <= 1'b0;
                end
            end
            // PUSH path: source pushing AND output slot is busy this cycle
            // s_ready guarantees buf_valid is 0 when this fires
            if (s_valid && s_ready && !(m_ready || !m_valid)) begin
                buf_valid <= 1'b1;
                buf_data  <= s_data;
            end
        end
    end
endmodule""",
    },

    # ------------------------------------------------------------------
    "fsm": {
        "keywords": [
            "state machine", "fsm", "sequence detector",
            "protocol controller", "handshake controller",
        ],
        "notes": (
            "Two-always FSM (registered state + combinational next-state). "
            "LLM fills state encoding, number of states, and transition logic. "
            "Template is intentionally sparse — FSMs vary too much for a full skeleton."
        ),
        "invariants": [
            r"always\s*@\s*\(\s*posedge\s+clk\s+or\s+negedge\s+rst_n\s*\)",
            r"if\s*\(\s*!rst_n\s*\)",
            r"always\s*@\s*\(\s*\*\s*\)",   # combinational block
            r"case\s*\(.*state",
        ],
        "skeleton": """\
// ChipGPT Skeleton: two-always FSM with async active-low reset
// LLM: fill state encoding, transitions, outputs. Keep two-always structure.
module <MODULE_NAME> (
    input  clk,
    input  rst_n,
    // LLM: add input/output ports as required by spec
    output reg out  // LLM: replace with actual outputs
);
    // LLM: define state encoding as localparams
    // Example: localparam IDLE=2'b00, ACTIVE=2'b01, DONE=2'b10;
    reg [1:0] state_reg, next_state;

    // Sequential: state register with async reset
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) state_reg <= 2'b00;  // LLM: replace 2'b00 with IDLE state
        else        state_reg <= next_state;
    end

    // Combinational: next-state logic
    always @(*) begin
        next_state = state_reg;  // default: hold state
        case (state_reg)
            // LLM: fill transitions here
            default: next_state = state_reg;
        endcase
    end

    // LLM: add output logic here (Moore: based on state_reg only)
endmodule""",
    },

    # ------------------------------------------------------------------
    "fifo_sync": {
        "keywords": [
            "synchronous fifo", "sync fifo", "fifo",
        ],
        "notes": (
            "Synchronous FIFO with circular buffer, full/empty flags. "
            "Depth and width are fillable. Does NOT cover async FIFO (needs CDC). "
            "Does NOT cover first-word fall-through (FWFT) variant."
        ),
        "invariants": [
            r"always\s*@\s*\(\s*posedge\s+clk\s+or\s+negedge\s+rst_n\s*\)",
            r"if\s*\(\s*!rst_n\s*\)",
            r"assign\s+full\s*=",
            r"assign\s+empty\s*=",
        ],
        "skeleton": """\
// ChipGPT Skeleton: synchronous FIFO with circular buffer
// LLM: fill WIDTH, DEPTH. Keep pointer arithmetic and flag logic.
module <MODULE_NAME> #(
    parameter WIDTH = <WIDTH>,
    parameter DEPTH = 16,
    parameter ADDR_W = 4   // LLM: set to $clog2(DEPTH), e.g. 4 for DEPTH=16
) (
    input              clk,
    input              rst_n,
    input              wr_en,
    input  [WIDTH-1:0] wr_data,
    input              rd_en,
    output [WIDTH-1:0] rd_data,
    output             full,
    output             empty
);
    reg [WIDTH-1:0]  mem [0:DEPTH-1];
    reg [ADDR_W:0]   wr_ptr, rd_ptr;   // extra bit for full/empty distinguish

    assign full  = (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]) &&
                   (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]);
    assign empty = (wr_ptr == rd_ptr);
    assign rd_data = mem[rd_ptr[ADDR_W-1:0]];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
        end else begin
            if (wr_en && !full)  begin
                mem[wr_ptr[ADDR_W-1:0]] <= wr_data;
                wr_ptr <= wr_ptr + 1;
            end
            if (rd_en && !empty) begin
                rd_ptr <= rd_ptr + 1;
            end
        end
    end
endmodule""",
    },

}


def get_skeleton(design_type: str, module_name: str, width: int = 8) -> str:
    """Return the skeleton with placeholders filled.
    Returns empty string if design_type not in library.
    """
    entry = SKELETON_LIBRARY.get(design_type)
    if not entry:
        return ""
    s = entry["skeleton"]
    s = s.replace("<MODULE_NAME>", module_name)
    s = s.replace("<WIDTH>", str(width))
    return s


def get_invariants(design_type: str) -> List[str]:
    """Return the list of invariant regex patterns for a design type."""
    entry = SKELETON_LIBRARY.get(design_type)
    return entry["invariants"] if entry else []


def detect_design_type(spec: dict, prompt: str) -> str:
    """Classify design type from spec and prompt.
    Returns a key into SKELETON_LIBRARY or 'generic'.
    Checks multi-word patterns before single keywords to avoid false matches.
    """
    text = (spec.get("formalized_request", "") + " " + prompt).lower()

    # Check in priority order — all entries, longest keywords first
    # Build a flat list of (keyword, design_type) sorted by keyword length desc
    candidates = []
    for dtype, entry in SKELETON_LIBRARY.items():
        for kw in entry["keywords"]:
            candidates.append((len(kw), kw, dtype))
    candidates.sort(reverse=True)  # longest keyword first

    for _, kw, dtype in candidates:
        if kw in text:
            return dtype

    return "generic"


def check_skeleton_invariants(v_code: str, design_type: str) -> tuple:
    """Check that generated RTL preserves skeleton invariants.
    Returns (passed: bool, violations: list of str).
    Supports __fn: prefix for custom checkers that can't be expressed as simple regex.
    """
    import re
    invariants = get_invariants(design_type)
    if not invariants:
        return True, []

    violations = []
    for pattern in invariants:
        if pattern.startswith("__fn:"):
            fn_name = pattern[5:]
            ok, msg = _run_custom_invariant(fn_name, v_code)
            if not ok:
                violations.append(f"Custom check failed: {msg}")
        else:
            if not re.search(pattern, v_code, re.IGNORECASE | re.DOTALL):
                violations.append(f"Missing: {pattern}")

    return len(violations) == 0, violations


def _run_custom_invariant(fn_name: str, v_code: str) -> tuple:
    """Run a named custom invariant check. Returns (passed, message)."""
    import re

    if fn_name == "s_ready_uses_buf_valid_and_m_ready":
        # Find the assign s_ready = ... line
        m = re.search(r'assign\s+s_ready\s*=\s*([^\n;]+)', v_code, re.IGNORECASE)
        if not m:
            return False, "assign s_ready not found — s_ready must be a combinational assign"
        rhs = m.group(1).strip()
        # Must reference m_ready
        if "m_ready" not in rhs:
            return False, f"s_ready RHS missing m_ready: '{rhs[:40]}'"
        # Must have a negation (! or ~) of some signal — any register name is OK
        # This catches the broken case "assign s_ready = m_ready" (no buffer term)
        if not re.search(r'[!~]\s*\w', rhs):
            return False, f"s_ready RHS missing negated buffer term: '{rhs[:40]}'"
        return True, "ok"

    return True, f"unknown custom check '{fn_name}'"


def has_ready_valid_ports(port_interface: list) -> bool:
    """Return True if the port interface contains ready/valid handshake signals."""
    if not port_interface:
        return False
    names = {p.get("name", "").lower() for p in port_interface}
    return bool(names & {"s_valid", "m_valid", "s_ready", "m_ready", "valid", "ready"})


def build_protocol_assertions(port_interface: list, design_name: str) -> str:
    """Generate Verilog-2001 protocol assertions for ready/valid interfaces.
    Uses manual history registers — no $past(), no SystemVerilog required.
    Only called when has_ready_valid_ports() returns True.
    """
    return """\
// Protocol assertions — Verilog-2001, no $past needed
// Checks: m_data must not change while m_valid=1 and m_ready=0
reg [7:0] _prev_m_data;
reg       _prev_m_valid;
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin _prev_m_data <= 0; _prev_m_valid <= 0; end
    else begin
        if (_prev_m_valid && !m_ready && m_valid)
            if (m_data !== _prev_m_data)
                $display("ASSERTION FAILED: m_data changed while m_valid=1 and m_ready=0");
        _prev_m_data  <= m_data;
        _prev_m_valid <= m_valid;
    end
end"""
