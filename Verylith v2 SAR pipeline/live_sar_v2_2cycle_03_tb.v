`timescale 1ns/1ps

module tb_live_sar_v2_2cycle_03;
  // SPECIFICATION: contract_version=sar-controller-contract-v2 contract_hash=d7af02003bf7d54aedb13c674860e0c1262076be9138e5ecc815328fbae7567f
  // ORACLE CONFIG: resolution=12 settle_cycles=2 comparator_high_keeps_bit=true sample_event=posedge clk
  localparam RESOLUTION = 12;
  localparam SETTLE_CYCLES = 2;
  localparam CMP_HIGH_KEEPS_BIT = 1;
  reg clk;
  reg rst_n;
  reg start;
  reg comparator;
  wire [RESOLUTION-1:0] dac_code;
  wire [RESOLUTION-1:0] adc_out;
  wire busy;
  wire done;
  reg [RESOLUTION-1:0] model_code;
  reg [RESOLUTION-1:0] trial_code;
  reg [RESOLUTION-1:0] expected_code;
  reg decision_keep;
  integer bit_index;
  integer settle_index;
  // Observational diagnostics only; no stimulus or assertions depend on the trace.
  integer edge_number = -1;
  integer accepted_edge;
  integer trial_edge;
  integer trace_idx;
  reg [2*RESOLUTION+2:0] trace_samples [0:7];

  always @(posedge clk) begin
    edge_number = edge_number + 1;
    #0.001;
    trace_samples[edge_number % 8] = {dac_code, adc_out, comparator, busy, done};
  end

  live_sar_v2_2cycle_03 dut (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .comparator(comparator),
    .dac_code(dac_code),
    .adc_out(adc_out),
    .busy(busy),
    .done(done)
  );

  always #5 clk = ~clk;

  task run_conversion;
    input [RESOLUTION-1:0] target_code;
    begin
      model_code = 0;
      start = 1;
      @(posedge clk); #1;
      accepted_edge = edge_number;
      start = 0;
      if (busy !== 1 || done !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: busy must rise on accepted start");
        $finish;
      end

      for (bit_index = RESOLUTION - 1; bit_index >= 0; bit_index = bit_index - 1) begin
        trial_code = model_code | (1 << bit_index);
        wait (dac_code === trial_code);
        @(negedge clk);
        trial_edge = edge_number;
        if (dac_code !== trial_code || busy !== 1 || done !== 0) begin
          $display("FUNCTIONAL_FAILURE: FAIL: invalid trial/busy state at bit %0d", bit_index);
          $finish;
        end

        // Hold the opposite decision throughout settling. An early sample changes
        // DAC/result behavior and fails before the contract sample edge.
        comparator = 0;
        if (bit_index == RESOLUTION - 1) begin
          start = 1;
          @(posedge clk); #1;
          start = 0;
          if (busy !== 1 || done !== 0) begin
            $display("FUNCTIONAL_FAILURE: FAIL: start while busy restarted or completed conversion");
            $finish;
          end
        end
        for (settle_index = 0; settle_index < SETTLE_CYCLES; settle_index = settle_index + 1) begin
          @(negedge clk);
          if (dac_code !== trial_code || busy !== 1 || done !== 0) begin
            $display("FUNCTIONAL_FAILURE: FAIL: comparator decision consumed before %0d full settle cycles at bit %0d", SETTLE_CYCLES, bit_index);
            $finish;
          end
        end

        decision_keep = (trial_code <= target_code);
        comparator = CMP_HIGH_KEEPS_BIT ? decision_keep : !decision_keep;
        @(posedge clk); #1;
        expected_code = decision_keep ? trial_code : (trial_code & ~(1 << bit_index));
        if (dac_code !== expected_code || busy !== 1 || done !== 0) begin
          $display("FUNCTIONAL_FAILURE: FAIL: bit %0d decision/sample timing expected dac=%h got=%h busy=%b done=%b", bit_index, expected_code, dac_code, busy, done);
          $display("SAR_DECISION bit=%0d trial=%h expected=%h observed=%h comparator=%b trial_edge=C+%0d expected_decision=C+%0d checked_edge=C+%0d",
                   bit_index, trial_code, expected_code, dac_code, comparator,
                   trial_edge-accepted_edge, trial_edge-accepted_edge+SETTLE_CYCLES+1, edge_number-accepted_edge);
          for (trace_idx = edge_number-7; trace_idx <= edge_number; trace_idx = trace_idx+1)
            if (trace_idx >= accepted_edge)
              $display("SAR_TRACE edge=C+%0d dac=%h adc=%h comparator=%b busy=%b done=%b",
                       trace_idx-accepted_edge, trace_samples[trace_idx%8][2*RESOLUTION+2:RESOLUTION+3],
                       trace_samples[trace_idx%8][RESOLUTION+2:3], trace_samples[trace_idx%8][2],
                       trace_samples[trace_idx%8][1], trace_samples[trace_idx%8][0]);
          $finish;
        end
        model_code = expected_code;
      end

      if (adc_out !== model_code || dac_code !== model_code) begin
        $display("FUNCTIONAL_FAILURE: FAIL: final result expected=%h adc=%h dac=%h", model_code, adc_out, dac_code);
        $finish;
      end
      if (busy !== 1 || done !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: busy/done timing at final decision is incorrect");
        $finish;
      end
      @(posedge clk); #1;
      if (done !== 1 || busy !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: done must assert exactly one cycle after final decision");
        $finish;
      end
      @(posedge clk); #1;
      if (done !== 0 || busy !== 0) begin
        $display("FUNCTIONAL_FAILURE: FAIL: done must be exactly one cycle and busy must remain low");
        $finish;
      end
    end
  endtask

  initial begin
    clk = 0;
    rst_n = 0;
    start = 0;
    comparator = 0;
    model_code = 0;
    trial_code = 0;
    expected_code = 0;
    decision_keep = 0;
    bit_index = 0;
    settle_index = 0;

    repeat (2) @(negedge clk);
    rst_n = 1;
    @(negedge clk);
    if ($bits(dut.dac_code) != RESOLUTION || $bits(dut.adc_out) != RESOLUTION) begin
      $display("CONTRACT_VIOLATION: SAR code ports must match resolution=%0d", RESOLUTION);
      $finish;
    end
    if (dac_code !== 0 || adc_out !== 0 || busy !== 0 || done !== 0) begin
      $display("FUNCTIONAL_FAILURE: FAIL: SAR reset outputs are not deterministic");
      $finish;
    end

    run_conversion(12'd2560);
    run_conversion(12'd1536);
    $display("COVER_HIT:sar_msb_to_lsb_binary_search");
    $display("COVER_HIT:sar_settle_sample_timing");
    $display("COVER_HIT:sar_busy_start_and_repeat");
    $display("COVER_HIT:sar_done_one_cycle");
    $display("SIMULATION_SUCCESS");
    $finish;
  end

  initial begin
    #(1000 + RESOLUTION * (SETTLE_CYCLES + 12) * 20);
    $display("FUNCTIONAL_FAILURE: FAIL: SAR DUT did not complete within the contract timeout");
    $finish;
  end
endmodule