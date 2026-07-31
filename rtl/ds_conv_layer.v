// ds_conv_layer — depthwise-separable convolution layer.
//
// Two-phase architecture:
//   Phase 1 (DW): C_IN parallel dw_conv3x3 instances process the input.
//                 Results stored in an intermediate buffer (IMG_W*IMG_H*C_IN).
//   Phase 2 (PW): pw_conv1x1 reads from the buffer, produces C_OUT outputs
//                 per spatial position.
//
// Input format: channel-interleaved pixel stream.
//   Pixels arrive one per clock cycling through channels:
//   ch0_pos0, ch1_pos0, ..., ch(C_IN-1)_pos0, ch0_pos1, ...
//
// This is a serial two-phase design — simple and correct.
// A pipelined version would overlap DW and PW, but requires
// backpressure or deep FIFOs.

module ds_conv_layer #(
    parameter IMG_W = 8,
    parameter IMG_H = 8,
    parameter C_IN  = 4,
    parameter C_OUT = 4
)(
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    input                    frame_start,
    output reg               out_valid,
    output reg signed [7:0]  pixel_out,
    output reg               out_last,
    // depthwise weights: 9 taps per input channel
    input  signed [7:0]      dw_weight [0:C_IN*9-1],
    input  signed [31:0]     dw_bias   [0:C_IN-1],
    input  signed [31:0]     dw_m0,
    input         [4:0]      dw_shift,
    // pointwise weights
    input  signed [7:0]      pw_weight [0:C_OUT*C_IN-1],
    input  signed [31:0]     pw_bias   [0:C_OUT-1],
    input  signed [31:0]     pw_m0,
    input         [4:0]      pw_shift
);

    localparam NUM_PX = IMG_W * IMG_H;

    // --- channel demux ---
    reg [$clog2(C_IN)-1:0] ch_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start)
            ch_cnt <= 0;
        else if (in_valid)
            ch_cnt <= (ch_cnt == C_IN - 1) ? 0 : ch_cnt + 1;
    end

    // --- C_IN parallel dw_conv3x3 instances ---
    wire [C_IN-1:0] dw_out_valid;
    wire signed [7:0] dw_out [0:C_IN-1];

    genvar ch;
    generate
        for (ch = 0; ch < C_IN; ch = ch + 1) begin : dw_inst
            wire ch_valid = in_valid && (ch_cnt == ch);

            wire signed [7:0] kw_local [0:8];
            genvar t;
            for (t = 0; t < 9; t = t + 1) begin : kw_wire
                assign kw_local[t] = dw_weight[ch*9 + t];
            end

            dw_conv3x3 #(
                .IMG_W(IMG_W),
                .IMG_H(IMG_H)
            ) u_dw (
                .clk(clk),
                .rst_n(rst_n),
                .in_valid(ch_valid),
                .pixel_in(pixel_in),
                .frame_start(frame_start),
                .out_valid(dw_out_valid[ch]),
                .pixel_out(dw_out[ch]),
                .kw(kw_local),
                .bias(dw_bias[ch]),
                .rq_m0(dw_m0),
                .rq_shift(dw_shift)
            );
        end
    endgenerate

    // --- intermediate buffer: store DW outputs ---
    // Layout: buf[pos * C_IN + ch] — channel-interleaved, same as output order
    reg signed [7:0] dw_buf [0:NUM_PX*C_IN-1];
    reg [$clog2(NUM_PX)-1:0] dw_wr_pos [0:C_IN-1]; // per-channel write position

    integer ci;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            for (ci = 0; ci < C_IN; ci = ci + 1)
                dw_wr_pos[ci] <= 0;
        end else begin
            for (ci = 0; ci < C_IN; ci = ci + 1) begin
                if (dw_out_valid[ci]) begin
                    dw_buf[dw_wr_pos[ci] * C_IN + ci] <= dw_out[ci];
                    dw_wr_pos[ci] <= dw_wr_pos[ci] + 1;
                end
            end
        end
    end

    // --- DW phase complete detection ---
    // All channels done when dw_wr_pos[0] reaches NUM_PX
    reg dw_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start)
            dw_done <= 1'b0;
        else if (dw_out_valid[0] && dw_wr_pos[0] == NUM_PX - 1)
            dw_done <= 1'b1;
    end

    // --- Phase 2: PW processing ---
    // State machine: feed C_IN values per position to pw_conv1x1,
    // wait for it to produce C_OUT outputs, then next position.

    localparam PW_IDLE    = 2'd0;
    localparam PW_FEED    = 2'd1;
    localparam PW_WAIT    = 2'd2;
    localparam PW_DONE    = 2'd3;

    reg [1:0] pw_state;
    reg [$clog2(NUM_PX)-1:0] pw_pos;
    reg [$clog2(C_IN)-1:0] pw_feed_ci;
    reg [$clog2(C_OUT)-1:0] pw_got_co;

    // pw_conv1x1 interface signals
    reg pw_in_valid;
    reg signed [7:0] pw_in_data;
    reg pw_in_last;
    wire pw_out_valid;
    wire signed [7:0] pw_out_data;
    wire pw_out_last;

    pw_conv1x1 #(
        .C_IN(C_IN),
        .C_OUT(C_OUT)
    ) u_pw (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pw_in_valid),
        .in_data(pw_in_data),
        .in_last(pw_in_last),
        .out_valid(pw_out_valid),
        .out_data(pw_out_data),
        .out_last(pw_out_last),
        .weight(pw_weight),
        .bias(pw_bias),
        .rq_m0(pw_m0),
        .rq_shift(pw_shift)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            pw_state    <= PW_IDLE;
            pw_pos      <= 0;
            pw_feed_ci  <= 0;
            pw_got_co   <= 0;
            pw_in_valid <= 1'b0;
            pw_in_data  <= 8'sd0;
            pw_in_last  <= 1'b0;
            out_valid   <= 1'b0;
            pixel_out   <= 8'sd0;
            out_last    <= 1'b0;
        end else begin
            pw_in_valid <= 1'b0;
            pw_in_last  <= 1'b0;
            out_valid   <= 1'b0;
            out_last    <= 1'b0;

            // forward pw outputs to module output
            if (pw_out_valid) begin
                out_valid <= 1'b1;
                pixel_out <= pw_out_data;
                out_last  <= pw_out_last;
            end

            case (pw_state)
                PW_IDLE: begin
                    if (dw_done) begin
                        pw_state   <= PW_FEED;
                        pw_pos     <= 0;
                        pw_feed_ci <= 0;
                    end
                end

                PW_FEED: begin
                    pw_in_valid <= 1'b1;
                    pw_in_data  <= dw_buf[pw_pos * C_IN + pw_feed_ci];
                    pw_in_last  <= (pw_feed_ci == C_IN - 1);

                    if (pw_feed_ci == C_IN - 1) begin
                        pw_feed_ci <= 0;
                        pw_state   <= PW_WAIT;
                        pw_got_co  <= 0;
                    end else begin
                        pw_feed_ci <= pw_feed_ci + 1;
                    end
                end

                PW_WAIT: begin
                    if (pw_out_valid) begin
                        if (pw_got_co == C_OUT - 1) begin
                            if (pw_pos == NUM_PX - 1) begin
                                pw_state <= PW_DONE;
                            end else begin
                                pw_pos   <= pw_pos + 1;
                                pw_state <= PW_FEED;
                            end
                        end else begin
                            pw_got_co <= pw_got_co + 1;
                        end
                    end
                end

                PW_DONE: begin
                    // stay here
                end
            endcase
        end
    end

endmodule
