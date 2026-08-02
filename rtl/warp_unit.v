// warp_unit — bilinear grid_sample (align_corners=1, border clamp) for VFI.
//
// Two-phase streaming:
//   Phase 1 (capture): IMG_W*IMG_H pixels streamed in row-major with in_valid.
//   Phase 2 (warp): one (flow_x, flow_y) per output pixel in row-major order;
//                   emits out_valid + pixel_out (INT8, round-half-up).
//
// Flow is displacement in pixels, fixed-point Q<FLOW_Q>:
//   src = out_pixel + flow   (flow=0 => identity)
// Matching ONNX GridSample: mode=bilinear, align_corners=1, padding=border.

module warp_unit #(
    parameter IMG_W  = 16,
    parameter IMG_H  = 16,
    parameter FLOW_Q = 8,
    parameter FLOW_W = 16
)(
    input                    clk,
    input                    rst_n,
    // frame capture
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    // flow stream (one per output pixel, row-major)
    input                    flow_valid,
    input  signed [FLOW_W-1:0] flow_x,
    input  signed [FLOW_W-1:0] flow_y,
    // output
    output reg               out_valid,
    output reg signed [7:0]  pixel_out
);

    localparam NUM_PX = IMG_W * IMG_H;
    localparam POS_W  = (NUM_PX <= 1) ? 1 : $clog2(NUM_PX);

    // wide internal coordinate width: position<<Q + flow must not overflow
    localparam SRC_W = FLOW_W + FLOW_Q + 1;

    localparam S_CAPTURE = 1'b0;
    localparam S_WARP    = 1'b1;

    reg        state;
    reg [POS_W-1:0] wr_pos;
    reg [POS_W-1:0] rd_pos;

    reg signed [7:0] frame_mem [0:NUM_PX-1];

    wire [POS_W-1:0] rd_col = rd_pos % IMG_W;
    wire [POS_W-1:0] rd_row = rd_pos / IMG_W;

    // source coordinate in Q(FLOW_Q): sx = rd_col*2^Q + flow_x
    wire signed [SRC_W-1:0] sx =
        {{(SRC_W - POS_W - FLOW_Q){1'b0}}, rd_col, {FLOW_Q{1'b0}}} +
        {{(SRC_W - FLOW_W){flow_x[FLOW_W-1]}}, flow_x};
    wire signed [SRC_W-1:0] sy =
        {{(SRC_W - POS_W - FLOW_Q){1'b0}}, rd_row, {FLOW_Q{1'b0}}} +
        {{(SRC_W - FLOW_W){flow_y[FLOW_W-1]}}, flow_y};

    wire signed [SRC_W-1:0] x0 = sx >>> FLOW_Q;
    wire signed [SRC_W-1:0] y0 = sy >>> FLOW_Q;
    wire [FLOW_Q-1:0]        fx = sx[FLOW_Q-1:0];
    wire [FLOW_Q-1:0]        fy = sy[FLOW_Q-1:0];

    // border clamp: clamp raw floor AND raw floor+1 independently (matches ONNX)
    wire signed [SRC_W-1:0] x0c = (x0 > (IMG_W-1)) ? (IMG_W-1) : (x0 < 0 ? 0 : x0);
    wire signed [SRC_W-1:0] y0c = (y0 > (IMG_H-1)) ? (IMG_H-1) : (y0 < 0 ? 0 : y0);
    wire signed [SRC_W-1:0] x1  = x0 + 1;
    wire signed [SRC_W-1:0] y1  = y0 + 1;
    wire signed [SRC_W-1:0] x1c = (x1 > (IMG_W-1)) ? (IMG_W-1) : (x1 < 0 ? 0 : x1);
    wire signed [SRC_W-1:0] y1c = (y1 > (IMG_H-1)) ? (IMG_H-1) : (y1 < 0 ? 0 : y1);

    wire [POS_W-1:0] addr00 = y0c[POS_W-1:0] * IMG_W + x0c[POS_W-1:0];
    wire [POS_W-1:0] addr01 = y0c[POS_W-1:0] * IMG_W + x1c[POS_W-1:0];
    wire [POS_W-1:0] addr10 = y1c[POS_W-1:0] * IMG_W + x0c[POS_W-1:0];
    wire [POS_W-1:0] addr11 = y1c[POS_W-1:0] * IMG_W + x1c[POS_W-1:0];

    wire signed [7:0] p00 = frame_mem[addr00];
    wire signed [7:0] p01 = frame_mem[addr01];
    wire signed [7:0] p10 = frame_mem[addr10];
    wire signed [7:0] p11 = frame_mem[addr11];

    // bilinear in fixed point, round-half-up, clamp INT8
    wire signed [FLOW_Q+1:0] wx = {2'b0, fx};
    wire signed [FLOW_Q+1:0] wy = {2'b0, fy};
    wire signed [FLOW_Q+1:0] inv = (1 << FLOW_Q);

    wire signed [24:0] top = p00 * (inv - wx) + p01 * wx;
    wire signed [24:0] bot = p10 * (inv - wx) + p11 * wx;
    wire signed [41:0] acc = top * (inv - wy) + bot * wy;

    wire signed [41:0] rnd_bit = (1 << (2*FLOW_Q - 1));
    wire signed [41:0] acc_r = acc + rnd_bit;
    wire signed [41:0] res = acc_r >>> (2*FLOW_Q);

    wire signed [7:0]  out_c = (res > 127)  ? 8'sd127 :
                               (res < -128) ? -8'sd128 : res[7:0];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state     <= S_CAPTURE;
            wr_pos    <= 0;
            rd_pos    <= 0;
            out_valid <= 1'b0;
            pixel_out <= 8'sd0;
        end else begin
            out_valid <= 1'b0;

            case (state)
                S_CAPTURE: begin
                    if (in_valid) begin
                        frame_mem[wr_pos] <= pixel_in;
                        if (wr_pos == NUM_PX - 1) begin
                            state  <= S_WARP;
                            rd_pos <= 0;
                        end else begin
                            wr_pos <= wr_pos + 1;
                        end
                    end
                end

                S_WARP: begin
                    if (flow_valid) begin
                        out_valid <= 1'b1;
                        pixel_out <= out_c;
                        if (rd_pos == NUM_PX - 1) begin
                            state  <= S_CAPTURE;
                            wr_pos <= 0;
                        end else begin
                            rd_pos <= rd_pos + 1;
                        end
                    end
                end
            endcase
        end
    end

endmodule