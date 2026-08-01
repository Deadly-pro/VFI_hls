// line_buffer_stream — Streaming 3x3 window extractor with SAME padding.
// Two-line FIFO architecture (not full-frame capture).
// Enables 1-pixel-in / 1-window-out throughput after initial latency.

module line_buffer_stream #(
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

    localparam STRIDE = 1;
    localparam KERNEL = 3;
    localparam HALF_K = KERNEL >> 1;  // 1

    // Two line buffers: each holds IMG_W pixels
    reg signed [7:0] line_buf_0 [0:IMG_W-1];
    reg signed [7:0] line_buf_1 [0:IMG_W-1];

    // Column counter and row counter
    reg [$clog2(IMG_W):0] col_cnt;
    reg [$clog2(IMG_H):0] row_cnt;

    // Input shift register for current row (3 pixels wide)
    reg signed [7:0] pix_sr [0:KERNEL-1];

    // Output position tracking
    reg [$clog2(IMG_W):0] out_col;
    reg [$clog2(IMG_H):0] out_row;
    reg                    frame_active;

    // Pipeline registers for 3x3 window
    reg signed [7:0] w0_r, w1_r, w2_r, w3_r, w4_r, w5_r, w6_r, w7_r, w8_r;
    reg              valid_pipe [0:2];

    // Input pixel shift register update
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            pix_sr[0] <= 8'sd0;
            pix_sr[1] <= 8'sd0;
            pix_sr[2] <= 8'sd0;
            col_cnt   <= 0;
            row_cnt   <= 0;
            frame_active <= 1'b1;
        end else if (frame_active && in_valid) begin
            // Shift in new pixel
            pix_sr[0] <= pix_sr[1];
            pix_sr[1] <= pix_sr[2];
            pix_sr[2] <= pixel_in;

            if (col_cnt == IMG_W - 1) begin
                col_cnt <= 0;
                if (row_cnt == IMG_H - 1) begin
                    row_cnt <= 0;
                    frame_active <= 1'b0;
                end else begin
                    row_cnt <= row_cnt + 1;
                end
            end else begin
                col_cnt <= col_cnt + 1;
            end
        end
    end

    // Line buffer write (delayed by 1 cycle to align with output)
    always @(posedge clk) begin
        if (in_valid) begin
            if (row_cnt < IMG_H - 1) begin
                line_buf_1[col_cnt] <= pix_sr[2];  // current pixel goes to line 1
            end
            if (row_cnt > 0) begin
                line_buf_0[col_cnt] <= line_buf_1[col_cnt];  // line 1 shifts to line 0
            end
        end
    end

    // Output column/row tracking (output starts after 1 row + 1 col latency)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            out_col <= 0;
            out_row <= 0;
            valid_pipe[0] <= 1'b0;
            valid_pipe[1] <= 1'b0;
            valid_pipe[2] <= 1'b0;
        end else begin
            valid_pipe[0] <= in_valid;
            valid_pipe[1] <= valid_pipe[0];
            valid_pipe[2] <= valid_pipe[1];

            if (valid_pipe[2]) begin
                if (out_col == IMG_W - 1) begin
                    out_col <= 0;
                    out_row <= out_row + 1;
                end else begin
                    out_col <= out_col + 1;
                end
            end
        end
    end

    // Combinational 3x3 window extraction with SAME padding (zero padding)
    wire at_top    = (out_row == 0);
    wire at_bottom = (out_row == IMG_H - 1);
    wire at_left   = (out_col == 0);
    wire at_right  = (out_col == IMG_W - 1);

    // Row -1 (top padding or line_buf_0)
    wire signed [7:0] r_m1_c_m1 = (at_top || at_left)   ? 8'sd0 : line_buf_0[out_col - 1];
    wire signed [7:0] r_m1_c_0  = at_top                 ? 8'sd0 : line_buf_0[out_col];
    wire signed [7:0] r_m1_c_p1 = (at_top || at_right)  ? 8'sd0 : line_buf_0[out_col + 1];

    // Row 0 (current row from shift register)
    wire signed [7:0] r_0_c_m1  = at_left  ? 8'sd0 : pix_sr[0];
    wire signed [7:0] r_0_c_0   = pix_sr[1];
    wire signed [7:0] r_0_c_p1  = at_right ? 8'sd0 : pix_sr[2];

    // Row +1 (line_buf_1 or bottom padding)
    wire signed [7:0] r_p1_c_m1 = (at_bottom || at_left)  ? 8'sd0 : line_buf_1[out_col - 1];
    wire signed [7:0] r_p1_c_0  = at_bottom                ? 8'sd0 : line_buf_1[out_col];
    wire signed [7:0] r_p1_c_p1 = (at_bottom || at_right) ? 8'sd0 : line_buf_1[out_col + 1];

    // Pipeline the window for timing
    always @(posedge clk) begin
        w0_r <= r_m1_c_m1; w1_r <= r_m1_c_0; w2_r <= r_m1_c_p1;
        w3_r <= r_0_c_m1;  w4_r <= r_0_c_0;  w5_r <= r_0_c_p1;
        w6_r <= r_p1_c_m1; w7_r <= r_p1_c_0; w8_r <= r_p1_c_p1;
        out_valid <= valid_pipe[2];
    end

    assign win_0 = w0_r; assign win_1 = w1_r; assign win_2 = w2_r;
    assign win_3 = w3_r; assign win_4 = w4_r; assign win_5 = w5_r;
    assign win_6 = w6_r; assign win_7 = w7_r; assign win_8 = w8_r;

endmodule