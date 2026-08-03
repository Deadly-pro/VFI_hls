// dw_conv3x3_synth — Synthesis wrapper with flat kw port (Yosys doesn't
// parse unpacked array ports). 9×8-bit kernel → 72-bit flat vector.
// Usage: read_verilog dw_conv3x3_synth.v dw_conv3x3.v line_buffer.v requantize.v

module dw_conv3x3_synth #(
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
    input  signed [71:0]     kw_flat,       // 9 × 8-bit kernel, row-major
    input  signed [31:0]     bias,
    input  signed [31:0]     rq_m0,
    input         [4:0]      rq_shift
);

    wire signed [7:0] kw [0:8];
    genvar i;
    generate
        for (i = 0; i < 9; i = i + 1) begin : kw_unpack
            assign kw[i] = kw_flat[i*8 +: 8];
        end
    endgenerate

    dw_conv3x3 #(
        .IMG_W(IMG_W),
        .IMG_H(IMG_H)
    ) u (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .pixel_in(pixel_in),
        .frame_start(frame_start),
        .out_valid(out_valid),
        .pixel_out(pixel_out),
        .kw(kw),
        .bias(bias),
        .rq_m0(rq_m0),
        .rq_shift(rq_shift)
    );

endmodule
