// dw_conv3x3 — depthwise 3x3 convolution for a single channel.
//
// Wires: line_buffer → 9-element dot product + bias → requantize → INT8 out.
//
// Streams one pixel per clock.  The 3x3 kernel weights and bias are held in
// registers (loaded externally before inference starts).  The requantize
// parameters (m0, shift) are also register-held.
//
// Total pipeline latency: line_buffer (2 cyc) + MAC (1 cyc) + requantize (2 cyc) = 5 cyc.

module dw_conv3x3 #(
    parameter IMG_W = 8,
    parameter IMG_H = 8
)(
    input                    clk,
    input                    rst_n,
    // pixel streaming
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    input                    frame_start,
    output                   out_valid,
    output signed [7:0]      pixel_out,
    // kernel weights (load before inference)
    input  signed [7:0]      kw [0:8],  // 3x3 kernel, row-major
    input  signed [31:0]     bias,
    // requantize params
    input  signed [31:0]     rq_m0,
    input         [4:0]      rq_shift
);

    // --- line buffer: extract 3x3 window ---
    wire               lb_valid;
    wire signed [7:0]  w0, w1, w2, w3, w4, w5, w6, w7, w8;

    line_buffer #(
        .IMG_W(IMG_W),
        .IMG_H(IMG_H)
    ) u_lb (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .pixel_in(pixel_in),
        .frame_start(frame_start),
        .out_valid(lb_valid),
        .win_0(w0), .win_1(w1), .win_2(w2),
        .win_3(w3), .win_4(w4), .win_5(w5),
        .win_6(w6), .win_7(w7), .win_8(w8)
    );

    // --- 9-element dot product + bias (combinational, then registered) ---
    wire signed [15:0] p0 = w0 * kw[0];
    wire signed [15:0] p1 = w1 * kw[1];
    wire signed [15:0] p2 = w2 * kw[2];
    wire signed [15:0] p3 = w3 * kw[3];
    wire signed [15:0] p4 = w4 * kw[4];
    wire signed [15:0] p5 = w5 * kw[5];
    wire signed [15:0] p6 = w6 * kw[6];
    wire signed [15:0] p7 = w7 * kw[7];
    wire signed [15:0] p8 = w8 * kw[8];

    wire signed [31:0] sum = bias
        + {{16{p0[15]}}, p0} + {{16{p1[15]}}, p1} + {{16{p2[15]}}, p2}
        + {{16{p3[15]}}, p3} + {{16{p4[15]}}, p4} + {{16{p5[15]}}, p5}
        + {{16{p6[15]}}, p6} + {{16{p7[15]}}, p7} + {{16{p8[15]}}, p8};

    // register the accumulator and valid
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

    // --- requantize ---
    requantize u_rq (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(acc_valid),
        .acc(acc_r),
        .m0(rq_m0),
        .shift(rq_shift),
        .out_valid(out_valid),
        .out(pixel_out)
    );

endmodule
