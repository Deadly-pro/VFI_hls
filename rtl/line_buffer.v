// line_buffer — 3x3 window extractor for SAME-padded convolution.
//
// Captures one complete IMG_W x IMG_H frame, then emits one zero-padded 3x3
// window per clock in row-major order.  Capturing the frame before emission
// makes every output position, including the bottom and right borders,
// available without requiring synthetic input pixels after the frame.

module line_buffer #(
    parameter IMG_W = 8,
    parameter IMG_H = 8
)(
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    input                    frame_start,
    output reg               out_valid,
    output signed [7:0]      win_0, win_1, win_2,
    output signed [7:0]      win_3, win_4, win_5,
    output signed [7:0]      win_6, win_7, win_8
);

    localparam NUM_PX = IMG_W * IMG_H;
    localparam POS_W  = (NUM_PX <= 1) ? 1 : $clog2(NUM_PX);

    localparam S_CAPTURE = 2'd0;
    localparam S_EMIT    = 2'd1;
    localparam S_DONE    = 2'd2;

    reg [1:0] state;
    reg [POS_W-1:0] write_pos;
    reg [POS_W-1:0] read_pos;
    reg [POS_W-1:0] out_pos;
    reg signed [7:0] frame_mem [0:NUM_PX-1];

    wire [POS_W-1:0] out_col = out_pos % IMG_W;
    wire [POS_W-1:0] out_row = out_pos / IMG_W;
    wire at_top    = (out_row == 0);
    wire at_bottom = (out_row == IMG_H - 1);
    wire at_left   = (out_col == 0);
    wire at_right  = (out_col == IMG_W - 1);

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            state     <= S_CAPTURE;
            write_pos <= 0;
            read_pos  <= 0;
            out_pos   <= 0;
            out_valid <= 1'b0;
        end else begin
            out_valid <= 1'b0;

            case (state)
                S_CAPTURE: begin
                    if (in_valid) begin
                        frame_mem[write_pos] <= pixel_in;
                        if (write_pos == NUM_PX - 1) begin
                            state    <= S_EMIT;
                            read_pos <= 0;
                        end else begin
                            write_pos <= write_pos + 1'b1;
                        end
                    end
                end

                S_EMIT: begin
                    out_valid <= 1'b1;
                    out_pos   <= read_pos;
                    if (read_pos == NUM_PX - 1) begin
                        state <= S_DONE;
                    end else begin
                        read_pos <= read_pos + 1'b1;
                    end
                end

                S_DONE: begin
                    // Wait for the next frame_start pulse.
                end

                default: state <= S_CAPTURE;
            endcase
        end
    end

    assign win_0 = (at_top || at_left)  ? 8'sd0 : frame_mem[out_pos - IMG_W - 1];
    assign win_1 = at_top                ? 8'sd0 : frame_mem[out_pos - IMG_W];
    assign win_2 = (at_top || at_right) ? 8'sd0 : frame_mem[out_pos - IMG_W + 1];
    assign win_3 = at_left               ? 8'sd0 : frame_mem[out_pos - 1];
    assign win_4 =                                  frame_mem[out_pos];
    assign win_5 = at_right              ? 8'sd0 : frame_mem[out_pos + 1];
    assign win_6 = (at_bottom || at_left)  ? 8'sd0 : frame_mem[out_pos + IMG_W - 1];
    assign win_7 = at_bottom                ? 8'sd0 : frame_mem[out_pos + IMG_W];
    assign win_8 = (at_bottom || at_right) ? 8'sd0 : frame_mem[out_pos + IMG_W + 1];

endmodule
