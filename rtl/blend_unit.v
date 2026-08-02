// blend_unit — mask-based synthesis: out = mask*a + (1-mask)*b.
//
// mask is Q0.8 unsigned [0,255]; a,b are INT8 warped frames.
//   out = (mask*a + (255-mask)*b + 127) >> 8  (round-half-up, clamp INT8)
// Matches the ONNX blend tail of nano_v16.onnx in fixed point.

module blend_unit (
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      in_a,
    input  signed [7:0]      in_b,
    input  [7:0]             mask,
    output reg               out_valid,
    output reg signed [7:0]  pixel_out
);

    // sign-extend a,b to 17 bits; mask is positive 9-bit
    wire signed [16:0] a17 = {{9{in_a[7]}}, in_a};
    wire signed [16:0] b17 = {{9{in_b[7]}}, in_b};
    wire signed [16:0] m9  = {1'b0, mask};

    // products fit in 25 bits; take low 17 (value-wise correct, sign carries)
    wire signed [16:0] ma = a17 * m9;
    wire signed [16:0] mb = b17 * (17'sd255 - m9);

    wire signed [16:0] acc = ma + mb + 17'sd127;
    wire signed [16:0] res = acc >>> 8;   // acc is signed => arithmetic shift

    wire signed [7:0]  out_c = (res > 127)  ? 8'sd127 :
                               (res < -128) ? -8'sd128 : res[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            pixel_out <= 8'sd0;
        end else begin
            out_valid <= in_valid;
            pixel_out <= out_c;
        end
    end

endmodule