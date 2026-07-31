// weight_mem — reloadable on-chip weight store (INT8).
//
// WHY THIS EXISTS
//   A hardwired accelerator bakes weights in as constants (frozen). To UPDATE
//   weights at runtime without re-synthesizing, they must live in a writable
//   memory. This module is that memory — the thing that makes the accelerator
//   field-updatable. Separating fixed datapath *structure* from updatable
//   *coefficients* is the line between hardwired and programmable accelerators.
//
// KEY DESIGN DECISION — read timing:
//   * async (combinational) read  -> flip-flops/LUTs  (tiny mems only, area blows up)
//   * sync  (registered)   read   -> real SRAM/block-RAM   <-- WE USE THIS
//   Registered read costs 1 cycle of latency (rd_data appears the cycle AFTER
//   rd_addr is presented). That latency is intentional; the conv pipeline will
//   be built to hide it.
//
// SIM GOTCHA: an unwritten `mem` reads as x (unknown), not 0. Rule: LOAD before
//   you READ (matches real usage: load all weights, then infer). The testbench
//   loads every address first for exactly this reason.
//
// DO NOT reset the mem array (real SRAMs don't reset — loading sets contents).

module weight_mem #(
    parameter DEPTH = 512,               // number of INT8 weights
    parameter AW    = 9                  // address width = clog2(DEPTH)
)(
    input                     clk,
    // --- weight-load / update port (runtime, AXI-Lite-ish) ---
    input                     wl_we,     // write enable
    input      [AW-1:0]       wl_addr,
    input  signed [7:0]       wl_data,
    // --- read port that feeds the MACs (SYNCHRONOUS, 1-cycle latency) ---
    input      [AW-1:0]       rd_addr,
    output reg signed [7:0]   rd_data
);

    reg signed [7:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (wl_we) mem[wl_addr] <= wl_data;
        rd_data <= mem[rd_addr];
    end

endmodule
