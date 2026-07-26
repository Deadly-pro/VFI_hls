// mac_int8 — INT8 x INT8 -> INT32 multiply-accumulate (the datapath atom).
//
// Reference implementation for Sprint 0 (sets the testbench conventions).
// Behaviour:
//   async active-low reset -> acc = 0
//   clear (sync, priority over en) -> acc = 0
//   en   (sync) -> acc = acc + a*b   (signed)
// acc is registered, so acc at cycle t reflects operations issued through t-1.

module mac_int8 (
    input                    clk,
    input                    rst_n,   // active-low async reset
    input                    en,      // accumulate this cycle
    input                    clear,   // sync clear (wins over en)
    input  signed [7:0]      a,       // INT8
    input  signed [7:0]      b,       // INT8
    output reg signed [31:0] acc      // INT32 running sum
);

    // 8x8 signed product: worst case -128*-128 = 16384, fits signed[15:0].
    wire signed [15:0] prod = a * b;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            acc <= 32'sd0;
        else if (clear)
            acc <= 32'sd0;
        else if (en)
            acc <= acc + prod;   // prod sign-extended to 32 bits
    end

endmodule
