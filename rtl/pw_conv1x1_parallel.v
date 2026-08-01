// pw_conv1x1_parallel — Parallel pointwise 1x1 convolution with MAC array.
// Sprint 3 upgrade: processes PARALLEL_CO output channels simultaneously.
// Each output channel has its own MAC accumulator, eliminating the C_OUT*C_IN serial loop.

module pw_conv1x1_parallel #(
    parameter C_IN        = 4,
    parameter C_OUT       = 4,
    parameter PARALLEL_CO = 2  // Number of output channels processed in parallel (must divide C_OUT)
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
    // weights and biases (register interface)
    input  signed [7:0]      weight [0:C_OUT*C_IN-1],
    input  signed [31:0]     bias   [0:C_OUT-1],
    // requant params
    input  signed [31:0]     rq_m0,
    input  [4:0]             rq_shift
);

    localparam CO_GROUPS = C_OUT / PARALLEL_CO;

    // Input buffer (shared across parallel MACs)
    reg signed [7:0] in_buf [0:C_IN-1];
    reg [$clog2(C_IN)-1:0] in_idx;

    // Parallel MAC state
    localparam S_IDLE    = 2'd0;
    localparam S_LATCH   = 2'd1;
    localparam S_MAC     = 2'd2;
    localparam S_REQ     = 2'd3;
    localparam S_OUTPUT  = 2'd4;

    reg [2:0] state;
    reg [$clog2(CO_GROUPS)-1:0] group_idx;
    reg [$clog2(C_IN)-1:0]      ci_idx;

    // Parallel accumulators (one per output channel in current group)
    reg signed [31:0] acc [0:PARALLEL_CO-1];

    // Parallel requantize (combinational, one cycle)
    wire signed [63:0] rq_wide [0:PARALLEL_CO-1];
    wire [5:0]         rq_total_shift = 6'd31 + {1'b0, rq_shift};
    wire signed [63:0] rq_round_bit = 64'sd1 <<< (rq_total_shift - 6'd1);
    wire signed [63:0] rq_shifted [0:PARALLEL_CO-1];
    wire signed [7:0]  rq_clamped [0:PARALLEL_CO-1];

    genvar p;
    generate
        for (p = 0; p < PARALLEL_CO; p = p + 1) begin : rq_gen
            assign rq_wide[p]    = acc[p] * rq_m0;
            assign rq_shifted[p] = (rq_wide[p] + rq_round_bit) >>> rq_total_shift;
            assign rq_clamped[p] =
                (rq_shifted[p] > 64'sd127)  ? 8'sd127 :
                (rq_shifted[p] < -64'sd128) ? -8'sd128 :
                rq_shifted[p][7:0];
        end
    endgenerate

    // Output mux state
    reg [$clog2(PARALLEL_CO)-1:0] out_mux_idx;
    reg                             output_phase;

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            in_idx      <= 0;
            group_idx   <= 0;
            ci_idx      <= 0;
            out_mux_idx <= 0;
            output_phase <= 1'b0;
            out_valid   <= 1'b0;
            out_data    <= 8'sd0;
            out_last    <= 1'b0;
            for (i = 0; i < C_IN; i = i + 1)
                in_buf[i] <= 8'sd0;
            for (i = 0; i < PARALLEL_CO; i = i + 1)
                acc[i] <= 32'sd0;
        end else begin
            out_valid <= 1'b0;
            out_last  <= 1'b0;

            case (state)
                S_IDLE: begin
                    in_idx    <= 0;
                    group_idx <= 0;
                    ci_idx    <= 0;
                    output_phase <= 1'b0;
                    if (in_valid) begin
                        in_buf[0] <= in_data;
                        in_idx <= 1;
                        state <= (C_IN == 1) ? S_MAC : S_LATCH;
                    end
                end

                S_LATCH: begin
                    if (in_valid) begin
                        in_buf[in_idx] <= in_data;
                        if (in_idx == C_IN - 1) begin
                            // Start first MAC group
                            state <= S_MAC;
                            ci_idx <= 0;
                            for (i = 0; i < PARALLEL_CO; i = i + 1)
                                acc[i] <= bias[group_idx * PARALLEL_CO + i];
                        end else begin
                            in_idx <= in_idx + 1;
                        end
                    end
                end

                S_MAC: begin
                    // Parallel MAC: all PARALLEL_CO channels accumulate in_buf[ci_idx] * weight
                    for (i = 0; i < PARALLEL_CO; i = i + 1) begin
                        integer co = group_idx * PARALLEL_CO + i;
                        integer w_idx = co * C_IN + ci_idx;
                        acc[i] <= acc[i] + in_buf[ci_idx] * weight[w_idx];
                    end

                    if (ci_idx == C_IN - 1) begin
                        state <= S_REQ;
                    end else begin
                        ci_idx <= ci_idx + 1;
                    end
                end

                S_REQ: begin
                    // Requantize all PARALLEL_CO channels (combinational, registered here)
                    state <= S_OUTPUT;
                    out_mux_idx <= 0;
                    output_phase <= 1'b1;
                end

                S_OUTPUT: begin
                    // Output PARALLEL_CO results sequentially
                    out_valid <= 1'b1;
                    out_data  <= rq_clamped[out_mux_idx];
                    out_last  <= (group_idx == CO_GROUPS - 1) && (out_mux_idx == PARALLEL_CO - 1);

                    if (out_mux_idx == PARALLEL_CO - 1) begin
                        if (group_idx == CO_GROUPS - 1) begin
                            // All output channels done
                            state <= S_IDLE;
                            output_phase <= 1'b0;
                        end else begin
                            // Next group
                            group_idx <= group_idx + 1;
                            ci_idx <= 0;
                            for (i = 0; i < PARALLEL_CO; i = i + 1)
                                acc[i] <= bias[(group_idx + 1) * PARALLEL_CO + i];
                            state <= S_MAC;
                        end
                        out_mux_idx <= 0;
                    end else begin
                        out_mux_idx <= out_mux_idx + 1;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule