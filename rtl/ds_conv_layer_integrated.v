// ds_conv_layer_integrated — depthwise-separable convolution layer with
// integrated reloadable weight memory (weight_mem).
//
// This wraps ds_conv_layer functionality but sources weights from
// weight_mem instances instead of register-array ports.
// Enables runtime weight loading via (wl_we, wl_addr, wl_data) without resynthesis.

module ds_conv_layer_integrated #(
    parameter IMG_W = 8,
    parameter IMG_H = 8,
    parameter C_IN  = 4,
    parameter C_OUT = 4,
    parameter DW_MEM_DEPTH = 512,
    parameter DW_MEM_AW    = 9,
    parameter PW_MEM_DEPTH = 512,
    parameter PW_MEM_AW    = 9
)(
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    input                    frame_start,
    output reg               out_valid,
    output reg signed [7:0]  pixel_out,
    output reg               out_last,

    // Weight-load interface (shared for DW and PW memories)
    input                    wl_we,
    input  [DW_MEM_AW-1:0]   wl_addr,
    input  signed [7:0]      wl_data,

    // Requantization parameters (registers, not in weight_mem)
    input  signed [31:0]     dw_m0,
    input  [4:0]             dw_shift,
    input  signed [31:0]     pw_m0,
    input  [4:0]             pw_shift
);

    localparam NUM_PX = IMG_W * IMG_H;

    // ---- DW weight_mem ----
    // Layout: [ch*9 + t] for ch in 0..C_IN-1, t in 0..8 (9 taps)
    // DW bias stored at offset C_IN*9 .. C_IN*9 + C_IN - 1
    localparam DW_KERNEL_OFFSET = 0;
    localparam DW_BIAS_OFFSET   = C_IN * 9;
    localparam DW_WEIGHT_DEPTH  = DW_BIAS_OFFSET + C_IN;

    wire signed [7:0] dw_mem_rdata;
    weight_mem #(
        .DEPTH(DW_MEM_DEPTH),
        .AW(DW_MEM_AW)
    ) u_dw_weight_mem (
        .clk(clk),
        .wl_we(wl_we),
        .wl_addr(wl_addr),
        .wl_data(wl_data),
        .rd_addr(dw_rd_addr),
        .rd_data(dw_mem_rdata)
    );

    // ---- PW weight_mem ----
    // Layout: [co*C_IN + ci] for co in 0..C_OUT-1, ci in 0..C_IN-1
    // PW bias stored at offset C_OUT*C_IN .. C_OUT*C_IN + C_OUT - 1
    localparam PW_KERNEL_OFFSET = 0;
    localparam PW_BIAS_OFFSET   = C_OUT * C_IN;
    localparam PW_WEIGHT_DEPTH  = PW_BIAS_OFFSET + C_OUT;

    wire signed [7:0] pw_mem_rdata;
    weight_mem #(
        .DEPTH(PW_MEM_DEPTH),
        .AW(PW_MEM_AW)
    ) u_pw_weight_mem (
        .clk(clk),
        .wl_we(wl_we),
        .wl_addr(wl_addr),
        .wl_data(wl_data),
        .rd_addr(pw_rd_addr),
        .rd_data(pw_mem_rdata)
    );

    // ---- Internal DW phase (same as ds_conv_layer but reading from weight_mem) ----
    // Channel demux
    reg [$clog2(C_IN)-1:0] ch_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start)
            ch_cnt <= 0;
        else if (in_valid)
            ch_cnt <= (ch_cnt == C_IN - 1) ? 0 : ch_cnt + 1;
    end

    // DW weight read address generator
    reg [DW_MEM_AW-1:0] dw_rd_addr;
    reg [3:0]           dw_kidx [0:C_IN-1]; // kernel index 0..8 per channel
    reg                 dw_bias_phase [0:C_IN-1];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            for (integer ci = 0; ci < C_IN; ci = ci + 1) begin
                dw_kidx[ci] <= 0;
                dw_bias_phase[ci] <= 1'b0;
            end
            dw_rd_addr <= DW_KERNEL_OFFSET;
        end else begin
            // Read kernel weights sequentially during first cycles
            if (dw_rd_addr < DW_KERNEL_OFFSET + C_IN * 9) begin
                dw_rd_addr <= dw_rd_addr + 1;
            end else if (dw_rd_addr < DW_WEIGHT_DEPTH) begin
                dw_rd_addr <= dw_rd_addr + 1;
            end
        end
    end

    // Kernel registers for each channel
    reg signed [7:0] dw_kernels [0:C_IN-1][0:8];
    reg signed [31:0] dw_biases [0:C_IN-1];

    integer ki;
    always @(posedge clk) begin
        if (wl_we && wl_addr >= DW_KERNEL_OFFSET && wl_addr < DW_KERNEL_OFFSET + C_IN * 9) begin
            integer ch_idx = (wl_addr - DW_KERNEL_OFFSET) / 9;
            integer tap_idx = (wl_addr - DW_KERNEL_OFFSET) % 9;
            dw_kernels[ch_idx][tap_idx] <= wl_data;
        end
        if (wl_we && wl_addr >= DW_BIAS_OFFSET && wl_addr < DW_WEIGHT_DEPTH) begin
            integer ch_idx = wl_addr - DW_BIAS_OFFSET;
            dw_biases[ch_idx] <= {{24{wl_data[7]}}, wl_data}; // sign-extend to 32-bit
        end
    end

    // DW instances
    wire [C_IN-1:0] dw_out_valid;
    wire signed [7:0] dw_out [0:C_IN-1];

    genvar ch;
    generate
        for (ch = 0; ch < C_IN; ch = ch + 1) begin : dw_inst
            wire ch_valid = in_valid && (ch_cnt == ch);

            wire signed [7:0] kw_local [0:8];
            genvar t;
            for (t = 0; t < 9; t = t + 1) begin : kw_wire
                assign kw_local[t] = dw_kernels[ch][t];
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
                .bias(dw_biases[ch]),
                .rq_m0(dw_m0),
                .rq_shift(dw_shift)
            );
        end
    endgenerate

    // ---- Intermediate buffer: store DW outputs ----
    reg signed [7:0] dw_buf [0:NUM_PX*C_IN-1];
    reg [$clog2(NUM_PX)-1:0] dw_wr_pos [0:C_IN-1];

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

    // DW phase complete detection
    reg dw_done;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start)
            dw_done <= 1'b0;
        else if (dw_out_valid[0] && dw_wr_pos[0] == NUM_PX - 1)
            dw_done <= 1'b1;
    end

    // ---- PW weight read address generator ----
    reg [PW_MEM_AW-1:0] pw_rd_addr;
    reg signed [7:0] pw_kernels [0:C_OUT-1][0:C_IN-1];
    reg signed [31:0] pw_biases [0:C_OUT-1];

    always @(posedge clk) begin
        if (wl_we && wl_addr >= PW_KERNEL_OFFSET && wl_addr < PW_KERNEL_OFFSET + C_OUT * C_IN) begin
            integer idx = wl_addr - PW_KERNEL_OFFSET;
            integer co = idx / C_IN;
            integer ci_idx = idx % C_IN;
            pw_kernels[co][ci_idx] <= wl_data;
        end
        if (wl_we && wl_addr >= PW_BIAS_OFFSET && wl_addr < PW_WEIGHT_DEPTH) begin
            integer co = wl_addr - PW_BIAS_OFFSET;
            pw_biases[co] <= {{24{wl_data[7]}}, wl_data};
        end
    end

    // ---- Phase 2: PW processing ----
    localparam PW_IDLE    = 2'd0;
    localparam PW_FEED    = 2'd1;
    localparam PW_WAIT    = 2'd2;
    localparam PW_DONE    = 2'd3;

    reg [1:0] pw_state;
    reg [$clog2(NUM_PX)-1:0] pw_pos;
    reg [$clog2(C_IN)-1:0] pw_feed_ci;
    reg [$clog2(C_OUT)-1:0] pw_got_co;

    reg pw_in_valid;
    reg signed [7:0] pw_in_data;
    reg pw_in_last;
    wire pw_out_valid;
    wire signed [7:0] pw_out_data;
    wire pw_out_last;

    // Flatten kernels for pw_conv1x1 port
    wire signed [7:0] pw_weight_flat [0:C_OUT*C_IN-1];
    wire signed [31:0] pw_bias_flat [0:C_OUT-1];
    genvar co, ci2;
    generate
        for (co = 0; co < C_OUT; co = co + 1) begin : pw_weight_flat_gen
            for (ci2 = 0; ci2 < C_IN; ci2 = ci2 + 1) begin : pw_weight_flat_inner
                assign pw_weight_flat[co*C_IN + ci2] = pw_kernels[co][ci2];
            end
            assign pw_bias_flat[co] = pw_biases[co];
        end
    endgenerate

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
        .weight(pw_weight_flat),
        .bias(pw_bias_flat),
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
                    // stay here until next frame_start
                end
            endcase
        end
    end

endmodule