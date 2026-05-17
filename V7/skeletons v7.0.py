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

import re
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
            r"always\s*@\s*\(\s*posedge\s+\w+\s+or\s+negedge\s+\w+\s*\)",
            r"if\s*\(\s*!\w+\s*\)",
            r"<=\s*[0-9'\{]",   # reset to 0, 32'h0, 1'b0, or {WIDTH{...}}
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
            r"always\s*@\s*\(\s*posedge\s+\w+\s+or\s+negedge\s+\w+\s*\)",
            r"if\s*\(\s*!\w+\s*\)",
            r"<=\s*[0-9'\{]",
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
            r"always\s*@\s*\(\s*posedge\s+\w+\s+or\s+negedge\s+\w+\s*\)",
            # Active-low reset check
            r"if\s*\(\s*!\w+\s*\)",
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
            r"always\s*@\s*\(\s*posedge\s+\w+\s+or\s+negedge\s+\w+\s*\)",
            r"if\s*\(\s*!\w+\s*\)",
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
    "uart_tx": {
        "keywords": [
            "uart transmitter", "uart tx", "uart_tx", "serial transmitter",
            "uart", "baud rate transmitter",
        ],
        "notes": (
            "UART TX FSM: IDLE->START->DATA(8 bits)->STOP->IDLE. "
            "Uses TICKS_PER_BAUD baud counter and 3-bit bit_index. "
            "FROZEN: state names, bit_index width ([2:0]), tx_out idle/start/stop values."
        ),
        "invariants": [
            r"always\s*@\s*\(\s*posedge\s+\w+\s+or\s+negedge\s+\w+\s*\)",
            r"if\s*\(\s*!\w+\s*\)",
            r"localparam|parameter",
            r"<=\s*[01'\{]",
        ],
        "skeleton": """\
// ChipGPT Skeleton: UART TX FSM, async active-low reset
// FROZEN: state encoding names, bit_index width [2:0], tx_out values per state.
// LLM: fill TICKS_PER_BAUD default, baud_counter width, transition conditions.
// CRITICAL WIDTH RULES:
//   bit_index    MUST be [2:0] — 3 bits — to index tx_data_reg[7:0] (bits 0-7)
//   baud_counter MUST be wide enough for TICKS_PER_BAUD-1
//     e.g. TICKS_PER_BAUD=4 -> [2:0] suffices; TICKS_PER_BAUD=434 -> [8:0] needed
//   NEVER use a 4-bit counter to index an 8-bit register (Verilator WIDTH error)
module <MODULE_NAME> #(parameter TICKS_PER_BAUD = <WIDTH>) (
    input        clk,
    input        rst_n,
    input        tx_valid,
    input  [7:0] tx_data,
    output reg   tx_ready,
    output reg   tx_out
);
    localparam IDLE  = 2'd0;
    localparam START = 2'd1;
    localparam DATA  = 2'd2;
    localparam STOP  = 2'd3;

    reg [1:0] state;
    reg [7:0] tx_data_reg;
    reg [2:0] bit_index;      // FROZEN: exactly [2:0] — 3 bits to index tx_data_reg[7:0]
                               //   A 4-bit index causes Verilator: "requires 3 bit index, not 4"
    reg [31:0] baud_counter;  // 32 bits: safe for unknown TICKS_PER_BAUD values

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= IDLE;
            tx_out      <= 1'b1;
            tx_ready    <= 1'b1;
            bit_index   <= 3'd0;
            // LLM: reset baud_counter to 0 here
        end else begin
            case (state)
                IDLE: begin
                    tx_out   <= 1'b1;
                    tx_ready <= 1'b1;
                    if (tx_valid) begin
                        tx_data_reg <= tx_data;
                        tx_ready    <= 1'b0;
                        // LLM: reset baud_counter to 0 here
                        state       <= START;
                    end
                end
                START: begin
                    tx_out <= 1'b0;   // start bit is always 0
                    // LLM: when baud_counter == TICKS_PER_BAUD-1:
                    //   reset baud_counter; bit_index <= 0; state <= DATA;
                end
                DATA: begin
                    tx_out <= tx_data_reg[bit_index];  // LSB first
                    // LLM: when baud_counter == TICKS_PER_BAUD-1:
                    //   if bit_index == 7: reset counter; state <= STOP;
                    //   else: reset counter; bit_index <= bit_index + 1;
                end
                STOP: begin
                    tx_out <= 1'b1;   // stop bit is always 1
                    // STOP must last exactly TICKS_PER_BAUD cycles (same as START and DATA)
                    // LLM: when baud_counter == TICKS_PER_BAUD-1:
                    //   baud_counter reset; state <= IDLE; tx_ready <= 1'b1;
                    // NEVER transition STOP->IDLE without checking baud_counter — only
                    // 1-cycle STOP violates the UART frame format.
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule""",
    },

    # ------------------------------------------------------------------
    "async_fifo": {
        "keywords": [
            "asynchronous fifo", "async fifo", "dual clock fifo", "cdc fifo",
            "gray code pointers", "gray-code pointers",
        ],
        "notes": (
            "Asynchronous FIFO structural guide. Freezes the CDC essentials: "
            "binary pointers, Gray pointers, and two-flop pointer synchronizers "
            "in the opposite clock domain. Width/depth/port names remain adaptable."
        ),
        "invariants": [
            r"wr_clk",
            r"rd_clk",
            r"gray",
            r"sync",
            r"assign\s+full\s*=",
            r"assign\s+empty\s*=",
        ],
        "skeleton": """\
// ChipGPT Skeleton: asynchronous FIFO with Gray-code CDC
// FROZEN: binary pointers, Gray pointers, two-flop synchronizers per clock domain.
module <MODULE_NAME> #(
    parameter WIDTH = <WIDTH>,
    parameter DEPTH = 16,
    parameter ADDR_W = 4
) (
    input              wr_clk,
    input              wr_rst_n,
    input              wr_en,
    input  [WIDTH-1:0] wr_data,
    output             full,
    input              rd_clk,
    input              rd_rst_n,
    input              rd_en,
    output [WIDTH-1:0] rd_data,
    output             empty
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [ADDR_W:0] wr_bin, rd_bin;
    reg [ADDR_W:0] wr_gray, rd_gray;
    reg [ADDR_W:0] rd_gray_wclk1, rd_gray_wclk2;
    reg [ADDR_W:0] wr_gray_rclk1, wr_gray_rclk2;

    function [ADDR_W:0] bin2gray;
        input [ADDR_W:0] bin;
        begin
            bin2gray = (bin >> 1) ^ bin;
        end
    endfunction

    assign full = (bin2gray(wr_bin + 1) ==
                   {~rd_gray_wclk2[ADDR_W:ADDR_W-1], rd_gray_wclk2[ADDR_W-2:0]});
    assign empty = (rd_gray == wr_gray_rclk2);
    assign rd_data = mem[rd_bin[ADDR_W-1:0]];

    always @(posedge wr_clk or negedge wr_rst_n) begin
        if (!wr_rst_n) begin
            wr_bin <= 0; wr_gray <= 0; rd_gray_wclk1 <= 0; rd_gray_wclk2 <= 0;
        end else begin
            rd_gray_wclk1 <= rd_gray;
            rd_gray_wclk2 <= rd_gray_wclk1;
            if (wr_en && !full) begin
                mem[wr_bin[ADDR_W-1:0]] <= wr_data;
                wr_bin <= wr_bin + 1;
                wr_gray <= bin2gray(wr_bin + 1);
            end
        end
    end

    always @(posedge rd_clk or negedge rd_rst_n) begin
        if (!rd_rst_n) begin
            rd_bin <= 0; rd_gray <= 0; wr_gray_rclk1 <= 0; wr_gray_rclk2 <= 0;
        end else begin
            wr_gray_rclk1 <= wr_gray;
            wr_gray_rclk2 <= wr_gray_rclk1;
            if (rd_en && !empty) begin
                rd_bin <= rd_bin + 1;
                rd_gray <= bin2gray(rd_bin + 1);
            end
        end
    end
endmodule""",
    },

    # ------------------------------------------------------------------
    "axi_stream_arbiter": {
        "keywords": [
            "axi-stream round robin arbiter", "axi4-stream round robin arbiter",
            "axis round robin arbiter", "round-robin arbiter", "round robin arbiter",
        ],
        "notes": (
            "AXI-stream round-robin arbiter guide. Register the grant and hold it "
            "until TLAST handshakes to avoid combinational ready loops."
        ),
        "invariants": [
            r"grant",
            r"tlast",
            r"m_axis_tvalid",
            r"s_axis_tready",
            r"always\s*@\s*\(\s*posedge",
        ],
        "skeleton": """\
// ChipGPT Skeleton: 4-port AXI4-Stream round-robin arbiter
// FROZEN: grant is registered; switch grants only after selected TLAST handshake.
module <MODULE_NAME> #(parameter WIDTH = <WIDTH>) (
    input clk,
    input rst_n,
    input [3:0] s_axis_tvalid,
    input [4*WIDTH-1:0] s_axis_tdata_flat,
    input [3:0] s_axis_tlast,
    output [3:0] s_axis_tready,
    output m_axis_tvalid,
    output [WIDTH-1:0] m_axis_tdata,
    output m_axis_tlast,
    input m_axis_tready
);
    reg [1:0] grant;
    wire selected_valid = s_axis_tvalid[grant];
    wire selected_last  = s_axis_tlast[grant];
    wire selected_fire  = selected_valid && m_axis_tready;

    assign m_axis_tvalid = selected_valid;
    assign m_axis_tlast  = selected_last;
    assign m_axis_tdata  = s_axis_tdata_flat[grant*WIDTH +: WIDTH];
    assign s_axis_tready = m_axis_tready ? (4'b0001 << grant) : 4'b0000;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            grant <= 0;
        end else if (selected_fire && selected_last) begin
            // LLM: replace with priority search from grant+1 among valid inputs.
            grant <= grant + 1;
        end
    end
endmodule""",
    },

    # ------------------------------------------------------------------
    "rv32i_decoder": {
        "keywords": [
            "rv32i instruction decoder", "risc-v instruction decoder",
            "riscv instruction decoder", "rv32i decoder",
            "rv32i immediate generator", "risc-v immediate generator",
            "riscv immediate generator",
        ],
        "notes": (
            "Pure combinational RV32I field decoder and immediate generator. "
            "Freezes field extraction, opcode-based immediate classes, and "
            "safe combinational defaults for control outputs."
        ),
        "invariants": [
            r"__fn:rv32i_decoder_semantics",
            r"always\s*@\s*\(\s*\*\s*\)",
            r"default\s*:",
        ],
        "skeleton": """\
// ChipGPT Skeleton: RV32I instruction decoder + immediate generator
// FROZEN: field extraction slices and safe combinational defaults.
module <MODULE_NAME> (
    input [31:0] instr,
    output [6:0] opcode,
    output [4:0] rd,
    output [2:0] funct3,
    output [4:0] rs1,
    output [4:0] rs2,
    output [6:0] funct7,
    output reg [31:0] imm_out,
    output reg reg_write,
    output reg alu_src,
    output reg branch,
    output reg mem_read,
    output reg mem_write
);
    assign opcode = instr[6:0];
    assign rd     = instr[11:7];
    assign funct3 = instr[14:12];
    assign rs1    = instr[19:15];
    assign rs2    = instr[24:20];
    assign funct7 = instr[31:25];

    always @(*) begin
        imm_out   = 32'd0;
        reg_write = 1'b0;
        alu_src   = 1'b0;
        branch    = 1'b0;
        mem_read  = 1'b0;
        mem_write = 1'b0;

        case (opcode)
            7'b0110011: begin
                reg_write = 1'b1; // R-type
            end
            7'b0010011: begin
                imm_out   = {{20{instr[31]}}, instr[31:20]}; // I-type ALU
                reg_write = 1'b1;
                alu_src   = 1'b1;
            end
            7'b0000011: begin
                imm_out   = {{20{instr[31]}}, instr[31:20]}; // load
                reg_write = 1'b1;
                alu_src   = 1'b1;
                mem_read  = 1'b1;
            end
            7'b0100011: begin
                imm_out   = {{20{instr[31]}}, instr[31:25], instr[11:7]}; // S-type
                alu_src   = 1'b1;
                mem_write = 1'b1;
            end
            7'b1100011: begin
                imm_out = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0}; // B-type
                branch  = 1'b1;
            end
            7'b0110111, 7'b0010111: begin
                imm_out   = {instr[31:12], 12'b0}; // U-type
                reg_write = 1'b1;
                alu_src   = 1'b1;
            end
            7'b1101111: begin
                imm_out   = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0}; // J-type
                reg_write = 1'b1;
            end
            default: begin
                imm_out   = 32'd0;
                reg_write = 1'b0;
                alu_src   = 1'b0;
                branch    = 1'b0;
                mem_read  = 1'b0;
                mem_write = 1'b0;
            end
        endcase
    end
endmodule""",
    },

    # ------------------------------------------------------------------
    "rv32i_regfile": {
        "keywords": [
            "risc-v register file", "riscv register file", "rv32i register file",
            "32x32 register file", "x0 must always read 0", "x0 read 0",
        ],
        "notes": (
            "RV32I integer register file. Freezes x0 zero behavior, asynchronous "
            "read ports, and synchronous write port."
        ),
        "invariants": [
            r"__fn:rv32i_regfile_semantics",
            r"5'd0",
            r"assign\s+\w+.*==\s*5'd0",
            r"always\s*@\s*\(\s*posedge",
        ],
        "skeleton": """\
// ChipGPT Skeleton: RV32I 32x32 register file
// FROZEN: x0 reads as zero and writes to x0 are ignored.
module <MODULE_NAME> (
    input clk,
    input we,
    input [4:0] rs1_addr,
    input [4:0] rs2_addr,
    input [4:0] rd_addr,
    input [31:0] rd_wdata,
    output [31:0] rs1_rdata,
    output [31:0] rs2_rdata
);
    reg [31:0] regs [0:31];

    assign rs1_rdata = (rs1_addr == 5'd0) ? 32'd0 : regs[rs1_addr];
    assign rs2_rdata = (rs2_addr == 5'd0) ? 32'd0 : regs[rs2_addr];

    always @(posedge clk) begin
        if (we && rd_addr != 5'd0) begin
            regs[rd_addr] <= rd_wdata;
        end
    end
endmodule""",
    },

    # ------------------------------------------------------------------
    "sram_byte_en": {
        "keywords": [
            "byte-enable sram", "byte enable sram", "sram byte enable",
            "sram byte-enable", "byte-addressed data memory",
            "byte addressed data memory",
        ],
        "notes": (
            "Synchronous 32-bit word memory with byte write enables. This is a "
            "physical-contract skeleton: it preserves a BRAM/SRAM-inferable shape "
            "instead of allowing simulation-correct async reads or read-merge muxes."
        ),
        "invariants": [
            r"__fn:sram_byte_en_semantics",
            r"always\s*@\s*\(\s*posedge\s+\w+\s*\)",
        ],
        "skeleton": """\
// ChipGPT Skeleton: synchronous byte-enable SRAM / data memory
// FROZEN PHYSICAL CONTRACT:
//   - memory array is never reset
//   - no asynchronous/combinational memory reads
//   - byte-lane writes occur directly inside always @(posedge clk)
//   - rdata is a registered output
module <MODULE_NAME> #(
    parameter MEM_DEPTH = 1024,
    parameter ADDR_WIDTH = 10
) (
    input clk,
    input rst_n,
    input mem_read,
    input mem_write,
    input [31:0] addr,
    input [31:0] wdata,
    input [3:0] byte_en,
    output [31:0] rdata
);
    reg [31:0] mem [0:MEM_DEPTH-1];
    reg [31:0] rdata_reg;
    wire [ADDR_WIDTH-1:0] word_addr;

    assign word_addr = addr[ADDR_WIDTH+1:2];
    assign rdata = rdata_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            rdata_reg <= 32'd0;
        end else begin
            if (mem_write) begin
                if (byte_en[0]) mem[word_addr][7:0]   <= wdata[7:0];
                if (byte_en[1]) mem[word_addr][15:8]  <= wdata[15:8];
                if (byte_en[2]) mem[word_addr][23:16] <= wdata[23:16];
                if (byte_en[3]) mem[word_addr][31:24] <= wdata[31:24];
            end

            if (mem_read) begin
                rdata_reg <= mem[word_addr];
            end else begin
                rdata_reg <= 32'd0;
            end
        end
    end
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
            r"always\s*@\s*\(\s*posedge\s+\w+\s+or\s+negedge\s+\w+\s*\)",
            r"if\s*\(\s*!\w+\s*\)",
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


def _iter_simple_always_blocks(v_code: str):
    """Yield (header, block_text) for simple Verilog always blocks."""
    import re
    text = re.sub(r'/\*.*?\*/', '', v_code, flags=re.DOTALL)
    text = re.sub(r'//.*', '', text)
    for m in re.finditer(r'always\s*@\s*\((.*?)\)', text, re.IGNORECASE | re.DOTALL):
        header = m.group(1)
        begin_m = re.search(r'\bbegin\b', text[m.end():], re.IGNORECASE)
        if not begin_m:
            semi = text.find(';', m.end())
            end_pos = len(text) if semi == -1 else semi + 1
            yield header, text[m.start():end_pos]
            continue
        pos = m.end() + begin_m.start()
        depth = 0
        end_pos = len(text)
        for tok in re.finditer(r'\bbegin\b|\bend\b', text[pos:], re.IGNORECASE):
            word = tok.group(0).lower()
            depth += 1 if word == "begin" else -1
            if depth == 0:
                end_pos = pos + tok.end()
                break
        yield header, text[m.start():end_pos]


def _strip_verilog_comments(v_code: str) -> str:
    import re
    text = re.sub(r'/\*.*?\*/', '', v_code, flags=re.DOTALL)
    return re.sub(r'//.*', '', text)


def looks_like_hierarchical_integration(spec: dict = None, prompt: str = "") -> bool:
    """Detect prompts that wire existing blocks instead of designing one primitive."""
    spec = spec or {}
    text = (
        str(spec.get("formalized_request", "")) + " " +
        str(spec.get("protocol_invariants", "")) + " " +
        str(spec.get("required_internal_registers", "")) + " " +
        str(prompt)
    ).lower()

    explicit_integration = (
        "must instantiate" in text
        or "do not rewrite" in text
        or "verified sub-module" in text
        or "verified submodule" in text
        or "verified sub-modules" in text
        or "verified submodules" in text
        or "existing sub-module" in text
        or "existing submodule" in text
        or "reuse sub-module" in text
        or "reuse submodule" in text
    )
    if explicit_integration:
        return True

    hierarchy_action = re.search(
        r"\b(instantiat\w*|wire(?:s|d|ing)?|connect(?:s|ed|ing)?|"
        r"integrat\w*|compos\w*)\b",
        text,
    )
    hierarchy_object = re.search(
        r"\b(sub-?modules?|verified modules?|existing modules?|child modules?|"
        r"building blocks?|components?)\b",
        text,
    )
    if hierarchy_action and hierarchy_object:
        return True

    system_terms = (
        "cpu core", "processor core", "single-cycle cpu", "single cycle cpu",
        "risc-v core", "riscv core", "rv32i core", "soc", "system-on-chip",
    )
    if any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in system_terms):
        return True

    wrapper_terms = ("top-level", "top level", "top module", "wrapper", "integration")
    wrapper_context = (
        "instantiate", "wire", "connect", "submodule", "sub-module",
        "verified", "existing module", "child module",
    )
    return any(term in text for term in wrapper_terms) and any(
        term in text for term in wrapper_context
    )


def detect_design_type(spec: dict, prompt: str) -> str:
    """Classify design type from spec and prompt.
    Returns a key into SKELETON_LIBRARY, a composed/hierarchical type, or 'generic'.
    Checks multi-word patterns before single keywords to avoid false matches.

    Returns 'generic' for bridge/adapter/interface designs even if a peripheral
    keyword matches — these compositions span two protocols and no single skeleton
    fits them. Detected by presence of conflicting interface signals or keywords.
    """
    text = (spec.get("formalized_request", "") + " " + prompt).lower()

    if looks_like_hierarchical_integration(spec, prompt):
        return "hierarchical_top"

    # Protocol composition detection. These are not single primitive skeletons:
    # they are assembled from protocol obligations and need deterministic
    # composed oracles rather than a monolithic LLM testbench.
    axis_present = any(kw in text for kw in (
        "axi4-stream", "axi-stream", "axi stream", "axis",
        "s_axis", "m_axis", "tvalid", "tready", "tdata",
    ))
    uart_tx_present = (
        "uart" in text
        and any(kw in text for kw in (
            "tx", "transmit", "transmitter", "serial output",
            "serialize", "serialise",
        ))
    )
    if axis_present and uart_tx_present:
        return "axi_stream_uart_tx"

    sram_present = any(kw in text for kw in (
        "sram", "data memory", "byte-addressed memory", "byte addressed memory",
        "mem_depth", "memory storing", "storing 32-bit words",
    ))
    byte_enable_present = any(kw in text for kw in (
        "byte_en", "byte enable", "byte-enable", "byte mask", "write mask",
    ))
    if sram_present and byte_enable_present:
        return "sram_byte_en"

    if any(kw in text for kw in (
        "asynchronous fifo", "async fifo", "dual clock fifo", "cdc fifo",
        "gray code pointers", "gray-code pointers",
    )):
        return "async_fifo"

    if axis_present and any(kw in text for kw in (
        "round robin arbiter", "round-robin arbiter", "arbiter",
    )):
        return "axi_stream_arbiter"

    rv32_present = any(kw in text for kw in ("rv32", "risc-v", "riscv"))
    decoder_present = any(kw in text for kw in (
        "instruction decoder", "decode instruction", "immediate generator",
        "imm_out",
    ))
    if rv32_present and decoder_present:
        return "rv32i_decoder"

    if any(kw in text for kw in (
        "rv32i register file", "risc-v register file", "riscv register file",
        "32x32 register file", "x0 must always read 0", "x0 read 0",
    )):
        return "rv32i_regfile"

    # Bridge/adapter detection: if prompt mixes interface standards and no
    # composed oracle exists yet, avoid forcing a single peripheral skeleton.
    # Examples: "APB to SPI bridge", "AHB to I2C adapter".
    # Count how many distinct protocol indicators are present
    protocols_found = set()
    peripheral_keywords = {"uart", "spi", "i2c", "can", "usb"}
    bus_keywords = {"axi", "apb", "ahb", "wishbone", "axis", "tvalid", "aclk"}
    for kw in peripheral_keywords:
        if kw in text:
            protocols_found.add("peripheral")
    for kw in bus_keywords:
        if kw in text:
            protocols_found.add("bus")
    # If both a bus protocol AND a peripheral are mentioned, it's a bridge — use generic
    if len(protocols_found) >= 2:
        return "generic"
    # Also return generic for any explicit bridge/adapter language
    if any(ind in text for ind in ("bridge", "adapter", "aclk", "aresetn",
                                    "s_axis", "m_axis", "tvalid", "tready")):
        return "generic"

    # Check in priority order — all entries, longest keywords first
    candidates = []
    for dtype, entry in SKELETON_LIBRARY.items():
        for kw in entry["keywords"]:
            candidates.append((len(kw), kw, dtype))
    candidates.sort(reverse=True)

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
        rhs_norm = re.sub(r'\s+', '', rhs).lower()
        if "m_ready" not in rhs_norm:
            return False, (
                f"s_ready equation wrong: got 'assign s_ready = {rhs[:50]}'. "
                f"MUST include m_ready. Canonical: assign s_ready = !buf_valid || m_ready;"
            )
        has_negated_buf = bool(re.search(
            r'[!~]\s*(?:buf|buffer)\w*', rhs, re.IGNORECASE
        ))
        has_demorgan = bool(re.search(r'!\s*\(', rhs))
        if not has_negated_buf and not has_demorgan:
            return False, (
                f"s_ready equation wrong: got 'assign s_ready = {rhs[:50]}'. "
                f"MUST negate a buffer signal. Canonical: assign s_ready = !buf_valid || m_ready;"
            )
        return True, "ok"

    if fn_name == "rv32i_decoder_semantics":
        text = re.sub(r'\s+', '', v_code).lower()
        required_slices = [
            "[6:0]", "[11:7]", "[14:12]", "[19:15]", "[24:20]", "[31:25]",
        ]
        for sl in required_slices:
            if f"instr{sl}" not in text:
                return False, f"RV32I decoder must extract instr{sl}"

        required_opcodes = [
            "7'b0110011",  # R-type
            "7'b0010011",  # I-type ALU
            "7'b0000011",  # load
            "7'b0100011",  # store
            "7'b1100011",  # branch
            "7'b0110111",  # LUI
            "7'b1101111",  # JAL
        ]
        for opcode in required_opcodes:
            if opcode not in text:
                return False, f"RV32I decoder missing opcode case {opcode}"

        imm_patterns = [
            r"\{20\{instr\[31\]\}\}.*instr\[31:20\]",  # I-type sign extend
            r"instr\[31:25\].*instr\[11:7\]",          # S-type
            r"instr\[7\].*instr\[30:25\].*instr\[11:8\].*1'b0",  # B-type
            r"instr\[31:12\].*12'b0",                  # U-type
            r"instr\[19:12\].*instr\[20\].*instr\[30:21\].*1'b0",  # J-type
        ]
        for pattern in imm_patterns:
            if not re.search(pattern, text):
                return False, "RV32I decoder missing one immediate format"

        for signal in ("reg_write", "alu_src", "branch", "mem_read", "mem_write"):
            if signal not in text:
                return False, f"RV32I decoder missing control signal {signal}"
        return True, "ok"

    if fn_name == "rv32i_regfile_semantics":
        mem = re.search(
            r'\breg\s*\[\s*31\s*:\s*0\s*\]\s+([A-Za-z_]\w*)\s*'
            r'\[\s*(?:0\s*:\s*31|31\s*:\s*0)\s*\]',
            v_code,
            re.IGNORECASE,
        )
        if not mem:
            return False, "RV32I regfile must declare one 32-bit x 32-entry reg array"
        mem_name = mem.group(1)
        text = re.sub(r'\s+', '', v_code).lower()
        mem_l = mem_name.lower()
        read_zero = len(re.findall(
            rf'assign\w+=\([^;?]*==5\'d0\)\?32\'d0:{mem_l}\[[^\]]+\]',
            text,
            re.IGNORECASE,
        )) >= 2
        if not read_zero:
            return False, "both async read ports must return 32'd0 when address is 5'd0"
        write_ignore = bool(re.search(
            rf'(?:we|write\w*)&&[^;]*(?:rd_addr|waddr|write_addr)\s*!=\s*5\'d0[^;]*{mem_l}\[',
            text,
            re.IGNORECASE,
        ))
        if not write_ignore:
            return False, "writes to x0 must be ignored with write_addr != 5'd0 guard"
        return True, "ok"

    if fn_name == "sram_byte_en_semantics":
        clean_code = _strip_verilog_comments(v_code)
        mem = re.search(
            r'\breg\s*\[\s*31\s*:\s*0\s*\]\s+([A-Za-z_]\w*)\s*'
            r'\[\s*0\s*:\s*MEM_DEPTH\s*-\s*1\s*\]',
            clean_code,
            re.IGNORECASE,
        )
        if not mem:
            return False, "SRAM must declare reg [31:0] memory [0:MEM_DEPTH-1]"
        mem_name = mem.group(1)
        text = re.sub(r'\s+', '', clean_code).lower()
        mem_l = mem_name.lower()

        if re.search(rf'\bassign\w*=[^;]*{mem_l}\[', text):
            return False, "SRAM must not use asynchronous assign-based memory reads"
        if re.search(rf'\bwire(?:\[[^\]]+\])?\w*=[^;]*{mem_l}\[', text):
            return False, "SRAM must not use wire-initializer asynchronous memory reads"

        for _, block in _iter_simple_always_blocks(clean_code):
            block_l = re.sub(r'\s+', '', block).lower()
            if "@(*)" in block_l and f"{mem_l}[" in block_l:
                return False, "SRAM must not read memory in combinational always @(*) logic"

        has_indexed_loop = bool(re.search(
            rf'for\s*\([^;]+;\s*[^;]*(?:<|<=)\s*4[^;]*;[^)]*\).*'
            rf'if\s*\([^)]*byte_en\s*\[\s*([A-Za-z_]\w*)\s*\][^)]*\).*'
            rf'{mem_name}\s*\[[^\]]+\]\s*\[\s*(?:8\s*\*\s*\1|\1\s*\*\s*8)\s*\+:\s*8\s*\]\s*<=\s*'
            rf'wdata\s*\[\s*(?:8\s*\*\s*\1|\1\s*\*\s*8)\s*\+:\s*8\s*\]',
            clean_code,
            re.IGNORECASE | re.DOTALL,
        ))

        for bit, lane in enumerate(("7:0", "15:8", "23:16", "31:24")):
            lane_pat = lane.replace(":", r"\s*:\s*")
            has_guard = bool(re.search(
                rf'if\s*\([^)]*byte_en\s*\[\s*{bit}\s*\]'
                rf'(?:\s*(?:==|===)\s*(?:1\s*\'\s*b\s*1|1))?[^)]*\)',
                clean_code,
                re.IGNORECASE | re.DOTALL,
            ))
            if not has_guard and not has_indexed_loop:
                return False, f"SRAM missing byte_en[{bit}] write guard"
            has_lane_write = bool(re.search(
                rf'{mem_name}\s*\[[^\]]+\]\s*\[\s*{lane_pat}\s*\]\s*<=\s*wdata\s*\[\s*{lane_pat}\s*\]',
                clean_code,
                re.IGNORECASE | re.DOTALL,
            ))
            if not has_lane_write and not has_indexed_loop:
                return False, f"SRAM byte lane {lane} must be written directly from wdata"

        if not re.search(
            rf'\brdata\w*\s*<=\s*{mem_name}\s*\[[^\]]+\]',
            clean_code,
            re.IGNORECASE,
        ):
            return False, "SRAM read data must be registered from memory inside the clocked block"
        for _, block in _iter_simple_always_blocks(clean_code):
            if not re.search(r'posedge', block, re.IGNORECASE):
                continue
            reset_m = re.search(
                r'if\s*\(\s*!\s*\w+\s*\)\s*begin(?P<body>.*?)end\s*else',
                block,
                re.IGNORECASE | re.DOTALL,
            )
            if reset_m and re.search(rf'{mem_name}\s*\[', reset_m.group('body'), re.IGNORECASE):
                return False, "SRAM memory array must not be reset or cleared"
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
