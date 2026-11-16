// CANONICAL SAR CONTRACT: version=sar-controller-contract-v2 hash=d7af02003bf7d54aedb13c674860e0c1262076be9138e5ecc815328fbae7567f resolution=12 settle_cycles=2 comparator_high_keeps_bit=1
// VERYLITH_SAR_CONTRACT version=sar-controller-contract-v2 hash=d7af02003bf7d54aedb13c674860e0c1262076be9138e5ecc815328fbae7567f
// ChipGPT AMS Contract: synchronous SAR ADC digital controller.
// RTL_INVARIANT: dac_code, adc_out, busy, and done are registered; !(done && busy).
// NUMERICAL_INVARIANT: test each bit MSB-to-LSB exactly once with a binary-search trial code.
// PHYSICAL_ASSUMPTION: comparator is valid at the S_DECIDE sampling edge after
// SETTLE_CYCLES full clk periods; CDAC settling accuracy is outside RTL proof.
// Conversion transaction: IDLE accepts start -> INIT -> SET_TRIAL -> SETTLE ->
// DECIDE for each bit -> DONE for exactly one clk cycle -> IDLE. start is ignored
// outside IDLE. CMP_HIGH_KEEPS_BIT=1 means comparator=1 keeps the trial bit.
module live_sar_v2_2cycle_03 #(
    parameter RESOLUTION = 12,
    parameter SETTLE_CYCLES = 2,
    parameter CMP_HIGH_KEEPS_BIT = 1
) (
    input              clk,
    input              rst_n,
    input              start,
    input              comparator,
    output reg [11:0]  dac_code,
    output reg [11:0]  adc_out,
    output reg         busy,
    output reg         done
);
    localparam S_IDLE      = 0;
    localparam S_INIT      = 1;
    localparam S_SET_TRIAL = 2;
    localparam S_SETTLE    = 3;
    localparam S_DECIDE    = 4;
    localparam S_DONE      = 5;

    reg [2:0] state;
    reg [3:0] current_bit;
    reg [3:0] bit_counter;
    reg [3:0] settle_counter;

    wire [11:0] selected_mask;
    wire        comparator_keeps_bit;
    wire [11:0] resolved_code;

    assign selected_mask = (12'b1 << current_bit);
    assign comparator_keeps_bit = CMP_HIGH_KEEPS_BIT ? comparator : ~comparator;
    assign resolved_code = comparator_keeps_bit ? dac_code : (dac_code & ~selected_mask);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            current_bit    <= 0;
            bit_counter    <= 0;
            settle_counter <= 0;
            dac_code       <= 0;
            adc_out        <= 0;
            busy           <= 0;
            done           <= 0;
        end else begin
            done <= 0;
            case (state)
                S_IDLE: begin
                    busy <= 0;
                    if (start) begin
                        busy  <= 1;
                        state <= S_INIT;
                    end
                end

                S_INIT: begin
                    current_bit    <= RESOLUTION - 1;
                    bit_counter    <= RESOLUTION - 1;
                    settle_counter <= 0;
                    dac_code       <= 0;
                    busy           <= 1;
                    state          <= S_SET_TRIAL;
                end

                S_SET_TRIAL: begin
                    dac_code       <= dac_code | selected_mask;
                    settle_counter <= 0;
                    busy           <= 1;
                    state          <= S_SETTLE;
                end

                S_SETTLE: begin
                    busy <= 1;
                    if (settle_counter == SETTLE_CYCLES - 1) begin
                        state <= S_DECIDE;
                    end else begin
                        settle_counter <= settle_counter + 1;
                    end
                end

                S_DECIDE: begin
                    busy     <= 1;
                    dac_code <= resolved_code;
                    if (current_bit == 0) begin
                        adc_out <= resolved_code;
                        state   <= S_DONE;
                    end else begin
                        current_bit <= current_bit - 1;
                        bit_counter <= bit_counter - 1;
                        state       <= S_SET_TRIAL;
                    end
                end

                S_DONE: begin
                    busy  <= 0;
                    done  <= 1;
                    state <= S_IDLE;
                end

                default: begin
                    state          <= S_IDLE;
                    current_bit    <= 0;
                    bit_counter    <= 0;
                    settle_counter <= 0;
                    dac_code       <= 0;
                    adc_out        <= 0;
                    busy           <= 0;
                    done           <= 0;
                end
            endcase
        end
    end
endmodule