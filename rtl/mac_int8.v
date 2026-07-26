module mac_int8 (
    input                    clk,
    input                    rst_n,   // active-low async reset
    input                    en,      // accumulate this cycle
    input                    clear,   // sync clear (wins over en)
    input  signed [7:0]      a,       // INT8
    input  signed [7:0]      b,       // INT8
    output reg signed [31:0] acc      // INT32 running sum
);

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
