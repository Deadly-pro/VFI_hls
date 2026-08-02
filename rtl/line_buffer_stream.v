// line_buffer_stream — Streaming 3x3 window extractor with SAME padding.
//
// Three cyclic line buffers (each one full input row) hold the three window
// rows. An independent output counter runs one row + one column behind the
// input, so every window's corner pixel is already captured when the window is
// emitted — no wait-states, no full-frame capture, O(W) memory.
// Border windows are zero-padded (SAME), matching golden/line_buffer_ref.py.

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

    localparam W = IMG_W;
    localparam H = IMG_H;

    // three cyclic row buffers; row x lives in row_buf[x % 3]
    reg signed [7:0] row_buf [0:2][0:W-1];

    reg [$clog2(H)-1:0] r;      // current input row
    reg [$clog2(W)-1:0] c;      // current input column
    reg                 first;  // captured at least one pixel

    // output window position (lags input by one row + one col)
    reg [$clog2(H)-1:0] ov;
    reg [$clog2(W)-1:0] oc;
    reg                 emit;   // output counter running
    reg                 done;

    // ---- capture: store each input pixel into its row's buffer ----
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            r     <= 0;
            c     <= 0;
            first <= 1'b0;
        end else if (in_valid) begin
            first <= 1'b1;
            if (c == W - 1) begin
                c <= 0;
                if (r == H - 1) r <= 0; else r <= r + 1;
            end else begin
                c <= c + 1;
            end
        end
    end

    // row_buf needs no reset (only valid columns are ever read), so keep it in
    // a clock-only block — an async-reset block would make Yosys's memory→
    // registers pass emit "Multiple edge sensitive events".
    always @(posedge clk) begin
        if (in_valid)
            row_buf[r % 3][c] <= pixel_in;
    end

    // ---- output counter: start one cycle after the first full corner is
    // captured (input at (1,1)), then run row-major until all H*W windows ----
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            ov <= 0; oc <= 0; emit <= 1'b0; done <= 1'b0;
        end else begin
            if (emit && !done) begin
                if (oc == W - 1) begin
                    oc <= 0;
                    if (ov == H - 1) done <= 1'b1; else ov <= ov + 1;
                end else begin
                    oc <= oc + 1;
                end
            end
            if (in_valid && r == 1 && c == 1)
                emit <= 1'b1;   // starts emitting on the next cycle
        end
    end

    // ---- generic SAME-padded 3x3 reader at window center (ov, oc) ----
    wire signed [7:0] t0 = (ov == 0 || oc == 0)        ? 8'sd0 : row_buf[(ov-1) % 3][oc-1];
    wire signed [7:0] t1 = (ov == 0)                   ? 8'sd0 : row_buf[(ov-1) % 3][oc];
    wire signed [7:0] t2 = (ov == 0 || oc == W-1)      ? 8'sd0 : row_buf[(ov-1) % 3][oc+1];
    wire signed [7:0] m0 = (oc == 0)                   ? 8'sd0 : row_buf[ov % 3][oc-1];
    wire signed [7:0] m1 =                              row_buf[ov % 3][oc];
    wire signed [7:0] m2 = (oc == W-1)                 ? 8'sd0 : row_buf[ov % 3][oc+1];
    wire signed [7:0] b0 = (ov == H-1 || oc == 0)      ? 8'sd0 : row_buf[(ov+1) % 3][oc-1];
    wire signed [7:0] b1 = (ov == H-1)                 ? 8'sd0 : row_buf[(ov+1) % 3][oc];
    wire signed [7:0] b2 = (ov == H-1 || oc == W-1)    ? 8'sd0 : row_buf[(ov+1) % 3][oc+1];

    // register the window (output counter moves on the same edge).
    // ponytail: synchronous reset — an async-reset block here made Yosys's
    // PROC_DFF emit "Multiple edge sensitive events" on out_valid.
    reg signed [7:0] w0_r, w1_r, w2_r, w3_r, w4_r, w5_r, w6_r, w7_r, w8_r;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            out_valid <= 1'b0;
            w0_r <= 0; w1_r <= 0; w2_r <= 0;
            w3_r <= 0; w4_r <= 0; w5_r <= 0;
            w6_r <= 0; w7_r <= 0; w8_r <= 0;
        end else begin
            out_valid <= emit && !done;
            if (emit && !done) begin
                w0_r <= t0; w1_r <= t1; w2_r <= t2;
                w3_r <= m0; w4_r <= m1; w5_r <= m2;
                w6_r <= b0; w7_r <= b1; w8_r <= b2;
            end
        end
    end

    assign win_0 = w0_r; assign win_1 = w1_r; assign win_2 = w2_r;
    assign win_3 = w3_r; assign win_4 = w4_r; assign win_5 = w5_r;
    assign win_6 = w6_r; assign win_7 = w7_r; assign win_8 = w8_r;

endmodule
