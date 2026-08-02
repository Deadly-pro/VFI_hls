// demo_rtl_tb — plain-Verilog driver for vfi_synth that reads input files and
// writes the interpolated frame. Used by tools/vfi_demo.py --rtl.
//
// Input files (decimal, one value per line):
//   weights/demo_t_i8.txt     t frame, channel-interleaved [H*W*3]
//   weights/demo_t1_i8.txt    t+1 frame, channel-interleaved
//   weights/demo_flowx.txt    flow_x per pixel (signed 16-bit) [H*W]
//   weights/demo_flowy.txt    flow_y per pixel (signed 16-bit) [H*W]
//   weights/demo_mask.txt     mask per pixel (uint8) [H*W]
// Output:
//   weights/demo_out.txt      interpolated frame, channel-interleaved INT8

`timescale 1ns/1ps

module demo_rtl_tb;

    parameter IMG_W = 64;
    parameter IMG_H = 64;
    parameter NCH   = 3;

    localparam NUM_PX = IMG_W * IMG_H;

    reg clk = 0;
    reg rst_n = 0;
    reg capture_valid = 0;
    reg signed [7:0] pixel_t_in = 0;
    reg signed [7:0] pixel_t1_in = 0;
    reg wb_valid = 0;
    reg signed [15:0] flow_x = 0;
    reg signed [15:0] flow_y = 0;
    reg [7:0] mask = 0;
    wire out_valid;
    wire signed [7:0] pixel_out;

    reg signed [7:0] t_i8   [0:NUM_PX*NCH-1];
    reg signed [7:0] t1_i8  [0:NUM_PX*NCH-1];
    reg signed [15:0] flowx [0:NUM_PX-1];
    reg signed [15:0] flowy [0:NUM_PX-1];
    reg [7:0] mask_q       [0:NUM_PX-1];

    integer i;

    vfi_synth #(
        .IMG_W(IMG_W), .IMG_H(IMG_H), .FLOW_Q(8), .FLOW_W(16), .NCH(NCH)
    ) u_vfi (
        .clk(clk), .rst_n(rst_n),
        .capture_valid(capture_valid),
        .pixel_t_in(pixel_t_in),
        .pixel_t1_in(pixel_t1_in),
        .wb_valid(wb_valid),
        .flow_x(flow_x), .flow_y(flow_y),
        .mask(mask),
        .out_valid(out_valid), .pixel_out(pixel_out)
    );

    always #5 clk = ~clk;

    integer out_fp;

    initial begin
        $readmemh("weights/demo_t_i8.txt", t_i8);
        $readmemh("weights/demo_t1_i8.txt", t1_i8);
        $readmemh("weights/demo_flowx.txt", flowx);
        $readmemh("weights/demo_flowy.txt", flowy);
        $readmemh("weights/demo_mask.txt", mask_q);

        #20 rst_n = 1;

        // --- capture phase (channel-interleaved) ---
        capture_valid = 1;
        for (i = 0; i < NUM_PX * NCH; i = i + 1) begin
            pixel_t_in  = t_i8[i];
            pixel_t1_in = t1_i8[i];
            #10;
        end
        capture_valid = 0;
        #10;

        // --- warp+blend: one flow position per (NCH+2) cycles ---
        out_fp = $fopen("weights/demo_out.txt", "w");
        for (i = 0; i < NUM_PX; i = i + 1) begin
            wb_valid = 1;
            flow_x = flowx[i];
            flow_y = flowy[i];
            mask   = mask_q[i];
            #10;
            wb_valid = 0;
            repeat (NCH + 2) begin
                #10;
                if (out_valid)
                    $fwrite(out_fp, "%d\n", pixel_out);
            end
        end
        wb_valid = 0;
        // drain any remaining
        repeat (NCH + 2) begin
            #10;
            if (out_valid)
                $fwrite(out_fp, "%d\n", pixel_out);
        end
        $fclose(out_fp);
        $display("[demo] sim complete");
        $finish;
    end

endmodule