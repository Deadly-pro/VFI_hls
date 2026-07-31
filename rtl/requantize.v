// requantize — INT32 accumulator → INT8 output (TFLite-style fixed-point rescale).
//
// Pipeline: 2 cycles latency (register the multiply, then shift+clamp).
//   cycle 0: wide = acc * m0           (64-bit signed multiply, registered)
//   cycle 1: shift, round, clamp → out (combinational + output register)

module requantize (
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [31:0]     acc,
    input  signed [31:0]     m0,       // Q0.31 multiplier
    input         [4:0]      shift,    // 0–31 additional right-shift
    output reg               out_valid,
    output reg signed [7:0]  out
);

    // --- stage 1: registered multiply ---
    reg signed [63:0] wide;
    reg        [4:0]  shift_r;
    reg               valid_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wide    <= 64'sd0;
            shift_r <= 5'd0;
            valid_r <= 1'b0;
        end else begin
            wide    <= acc * m0;
            shift_r <= shift;
            valid_r <= in_valid;
        end
    end

    // --- stage 2: shift + round + clamp ---
    wire [5:0]       total_shift = 6'd31 + {1'b0, shift_r};
    wire signed [63:0] round_bit = 64'sd1 <<< (total_shift - 6'd1);
    wire signed [63:0] nudged    = wide + round_bit;
    wire signed [63:0] shifted   = nudged >>> total_shift;

    // clamp to [-128, 127]
    wire signed [7:0] clamped =
        (shifted > 64'sd127)  ? 8'sd127 :
        (shifted < -64'sd128) ? -8'sd128 :
        shifted[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out       <= 8'sd0;
            out_valid <= 1'b0;
        end else begin
            out       <= clamped;
            out_valid <= valid_r;
        end
    end

endmodule
