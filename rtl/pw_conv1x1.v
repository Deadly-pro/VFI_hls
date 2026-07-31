// pw_conv1x1 — pointwise 1x1 convolution (channel mixing).
//
// For each spatial position, accumulates in[ci] * weight[co][ci] + bias[co]
// over all input channels, then requantizes to INT8.
//
// Operation: receives C_IN pixels per spatial position (one per clock),
// accumulates, then outputs C_OUT results (one per clock).
//
// Weight storage: weights are read from an external weight_mem.
// Weight layout in memory: weight[co * C_IN + ci] for (co, ci).
// Bias stored separately (C_OUT x INT32).
//
// For simplicity in this first version: weights and biases are held in
// registers (loaded before inference).  The MAC iterates over C_IN for
// each output channel, serialized.
//
// Data flow (per spatial position):
//   1. Latch C_IN input values (shift register)
//   2. For each co in [0, C_OUT):
//      a. Accumulate: acc = bias[co] + sum(input[ci] * weight[co][ci])
//      b. Requantize: out = requantize(acc, m0, shift)
//      c. Emit out with out_valid=1
//
// This is C_IN cycles to latch + C_OUT*C_IN cycles to compute = slow but correct.
// A MAC-array version parallelizes this; that's the Sprint 3 upgrade path.

module pw_conv1x1 #(
    parameter C_IN  = 4,
    parameter C_OUT = 4
)(
    input                    clk,
    input                    rst_n,
    // input stream: one channel value per clock, C_IN values per position
    input                    in_valid,
    input  signed [7:0]      in_data,
    input                    in_last,    // pulse on the last channel of this position
    // output stream
    output reg               out_valid,
    output reg signed [7:0]  out_data,
    output reg               out_last,   // pulse on last output channel
    // weights and biases (register interface, load before inference)
    input  signed [7:0]      weight [0:C_OUT*C_IN-1],
    input  signed [31:0]     bias   [0:C_OUT-1],
    // requantize params
    input  signed [31:0]     rq_m0,
    input         [4:0]      rq_shift
);

    // --- input latch ---
    reg signed [7:0] in_buf [0:C_IN-1];
    reg [$clog2(C_IN)-1:0] in_idx;
    reg input_ready;

    // --- compute state machine ---
    localparam S_IDLE    = 2'd0;
    localparam S_LATCH   = 2'd1;
    localparam S_COMPUTE = 2'd2;
    localparam S_OUTPUT  = 2'd3;

    reg [1:0] state;
    reg [$clog2(C_OUT)-1:0] co_idx;   // current output channel
    reg [$clog2(C_IN)-1:0]  ci_idx;   // current input channel in MAC loop
    reg signed [31:0] acc;

    // requantize (combinational — single-cycle version for the serial MAC)
    wire signed [63:0] rq_wide = acc * rq_m0;
    wire [5:0]         rq_total_shift = 6'd31 + {1'b0, rq_shift};
    wire signed [63:0] rq_round_bit = 64'sd1 <<< (rq_total_shift - 6'd1);
    wire signed [63:0] rq_shifted = (rq_wide + rq_round_bit) >>> rq_total_shift;
    wire signed [7:0]  rq_clamped =
        (rq_shifted > 64'sd127)  ? 8'sd127 :
        (rq_shifted < -64'sd128) ? -8'sd128 :
        rq_shifted[7:0];

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_IDLE;
            in_idx    <= 0;
            co_idx    <= 0;
            ci_idx    <= 0;
            acc       <= 32'sd0;
            out_valid <= 1'b0;
            out_data  <= 8'sd0;
            out_last  <= 1'b0;
            for (i = 0; i < C_IN; i = i + 1)
                in_buf[i] <= 8'sd0;
        end else begin
            out_valid <= 1'b0;
            out_last  <= 1'b0;

            case (state)
                S_IDLE: begin
                    in_idx <= 0;
                    if (in_valid) begin
                        in_buf[0] <= in_data;
                        in_idx <= 1;
                        state <= (C_IN == 1) ? S_COMPUTE : S_LATCH;
                        if (C_IN == 1) begin
                            co_idx <= 0;
                            ci_idx <= 0;
                            acc <= bias[0] + {{24{in_data[7]}}, in_data} * {{24{weight[0][7]}}, weight[0]};
                        end
                    end
                end

                S_LATCH: begin
                    if (in_valid) begin
                        in_buf[in_idx] <= in_data;
                        if (in_idx == C_IN - 1) begin
                            state  <= S_COMPUTE;
                            co_idx <= 0;
                            ci_idx <= 0;
                            acc    <= bias[0];
                        end else begin
                            in_idx <= in_idx + 1;
                        end
                    end
                end

                S_COMPUTE: begin
                    // MAC: acc += in_buf[ci_idx] * weight[co_idx * C_IN + ci_idx]
                    acc <= acc + in_buf[ci_idx] * weight[co_idx * C_IN + ci_idx];
                    if (ci_idx == C_IN - 1) begin
                        // done accumulating for this output channel
                        state <= S_OUTPUT;
                    end else begin
                        ci_idx <= ci_idx + 1;
                    end
                end

                S_OUTPUT: begin
                    // emit requantized result
                    out_valid <= 1'b1;
                    out_data  <= rq_clamped;
                    out_last  <= (co_idx == C_OUT - 1);

                    if (co_idx == C_OUT - 1) begin
                        // done with all output channels for this position
                        state <= S_IDLE;
                    end else begin
                        // start next output channel
                        co_idx <= co_idx + 1;
                        ci_idx <= 0;
                        acc    <= bias[co_idx + 1];
                        state  <= S_COMPUTE;
                    end
                end
            endcase
        end
    end

endmodule
