// leaf_flat.v — flat-port (Yosys-synthesizable) leaf datapath cells.
//
// Yosys read_verilog -sv cannot parse unpacked array ports, so the array-port
// leaf modules (dw_conv3x3, pw_conv1x1) are re-exposed here with flat bit-
// vector ports. The internal logic is inlined (no dependency on the original
// array-port files) so a top-level synth wrapper can be elaborated cleanly.

// --- depthwise 3x3 conv, flat 72-bit kernel (9 x 8-bit, row-major) ---
module dw_conv3x3_flat #(
    parameter IMG_W = 8,
    parameter IMG_H = 8
)(
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    input                    frame_start,
    output                   out_valid,
    output signed [7:0]      pixel_out,
    input  signed [71:0]     kw_flat,
    input  signed [31:0]     bias,
    input  signed [31:0]     rq_m0,
    input         [4:0]      rq_shift
);

    wire              lb_valid;
    wire signed [7:0] w0, w1, w2, w3, w4, w5, w6, w7, w8;

    line_buffer #(.IMG_W(IMG_W), .IMG_H(IMG_H)) u_lb(
        .clk(clk), .rst_n(rst_n), .in_valid(in_valid), .pixel_in(pixel_in),
        .frame_start(frame_start), .out_valid(lb_valid),
        .win_0(w0), .win_1(w1), .win_2(w2), .win_3(w3), .win_4(w4),
        .win_5(w5), .win_6(w6), .win_7(w7), .win_8(w8)
    );

    wire signed [7:0] k0 = kw_flat[7:0];
    wire signed [7:0] k1 = kw_flat[15:8];
    wire signed [7:0] k2 = kw_flat[23:16];
    wire signed [7:0] k3 = kw_flat[31:24];
    wire signed [7:0] k4 = kw_flat[39:32];
    wire signed [7:0] k5 = kw_flat[47:40];
    wire signed [7:0] k6 = kw_flat[55:48];
    wire signed [7:0] k7 = kw_flat[63:56];
    wire signed [7:0] k8 = kw_flat[71:64];

    wire signed [15:0] p0 = w0 * k0;
    wire signed [15:0] p1 = w1 * k1;
    wire signed [15:0] p2 = w2 * k2;
    wire signed [15:0] p3 = w3 * k3;
    wire signed [15:0] p4 = w4 * k4;
    wire signed [15:0] p5 = w5 * k5;
    wire signed [15:0] p6 = w6 * k6;
    wire signed [15:0] p7 = w7 * k7;
    wire signed [15:0] p8 = w8 * k8;

    wire signed [31:0] sum =
        bias
        + {{16{p0[15]}}, p0} + {{16{p1[15]}}, p1} + {{16{p2[15]}}, p2}
        + {{16{p3[15]}}, p3} + {{16{p4[15]}}, p4} + {{16{p5[15]}}, p5}
        + {{16{p6[15]}}, p6} + {{16{p7[15]}}, p7} + {{16{p8[15]}}, p8};

    reg signed [31:0] acc_r;
    reg               acc_valid;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_r     <= 32'sd0;
            acc_valid <= 1'b0;
        end else begin
            acc_r     <= sum;
            acc_valid <= lb_valid;
        end
    end

    requantize u_rq(
        .clk(clk), .rst_n(rst_n), .in_valid(acc_valid), .acc(acc_r),
        .m0(rq_m0), .shift(rq_shift), .out_valid(out_valid), .out(pixel_out)
    );
endmodule

// --- serial pointwise 1x1 conv, flat weight/bias ports ---
module pw_conv1x1_flat #(
    parameter C_IN  = 4,
    parameter C_OUT = 4
)(
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      in_data,
    input                    in_last,
    output reg               out_valid,
    output reg signed [7:0]  out_data,
    output reg               out_last,
    input  signed [C_OUT*C_IN*8-1:0] weight_flat,
    input  signed [C_OUT*32-1:0]     bias_flat,
    input  signed [31:0]     rq_m0,
    input         [4:0]      rq_shift
);

    localparam S_IDLE    = 0;
    localparam S_LATCH   = 1;
    localparam S_COMPUTE = 2;
    localparam S_OUTPUT  = 3;

    reg [1:0] state;
    reg [$clog2(C_IN)-1:0]  in_idx;
    reg [$clog2(C_OUT)-1:0] co_idx;
    reg [$clog2(C_IN)-1:0]  ci_idx;
    reg signed [31:0] acc;
    reg signed [7:0] in_buf [0:C_IN-1];

    wire signed [31:0] bias_sel = bias_flat[co_idx*32 +: 32];
    wire signed [7:0]  w_sel    = weight_flat[(co_idx*C_IN + ci_idx)*8 +: 8];

    wire signed [63:0] rq_wide = acc * rq_m0;
    wire [5:0] rq_total_shift = 6'd31 + {1'b0, rq_shift};
    wire signed [63:0] rq_round_bit = 64'sd1 <<< (rq_total_shift - 6'd1);
    wire signed [63:0] rq_shifted = (rq_wide + rq_round_bit) >>> rq_total_shift;
    wire signed [7:0]  rq_clamped =
        (rq_shifted > 64'sd127) ? 8'sd127 :
        (rq_shifted < -64'sd128) ? -8'sd128 : rq_shifted[7:0];

    integer i;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; in_idx <= 0; co_idx <= 0; ci_idx <= 0;
            acc <= 32'sd0; out_valid <= 1'b0; out_data <= 8'sd0; out_last <= 1'b0;
            for (i = 0; i < C_IN; i = i + 1) in_buf[i] <= 8'sd0;
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
                    end
                end
                S_LATCH: begin
                    if (in_valid) begin
                        in_buf[in_idx] <= in_data;
                        if (in_idx == C_IN - 1) begin
                            state <= S_COMPUTE; co_idx <= 0; ci_idx <= 0; acc <= bias_flat[0 +: 32];
                        end else in_idx <= in_idx + 1;
                    end
                end
                S_COMPUTE: begin
                    acc <= acc + in_buf[ci_idx] * w_sel;
                    if (ci_idx == C_IN - 1) state <= S_OUTPUT;
                    else ci_idx <= ci_idx + 1;
                end
                S_OUTPUT: begin
                    out_valid <= 1'b1;
                    out_data  <= rq_clamped;
                    out_last  <= (co_idx == C_OUT - 1);
                    if (co_idx == C_OUT - 1) state <= S_IDLE;
                    else begin
                        co_idx <= co_idx + 1; ci_idx <= 0;
                        acc <= bias_flat[(co_idx+1)*32 +: 32];
                        state <= S_COMPUTE;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule