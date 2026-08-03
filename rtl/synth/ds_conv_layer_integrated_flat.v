// ds_conv_layer_integrated_flat — Yosys-synthesizable version of
// ds_conv_layer_integrated with flat-port leaf cells.
//
// Same interface, same two-phase datapath (DW convs -> PW conv), weights
// loaded over the wl bus. The dead weight_mem instances are omitted. Leaf
// cells dw_conv3x3_flat / pw_conv1x1_flat come from leaf_flat.v, so the whole
// hierarchy has no unpacked array ports and synthesizes under
// `read_verilog -sv`.

module ds_conv_layer_integrated_flat #(
    parameter IMG_W = 8,
    parameter IMG_H = 8,
    parameter C_IN  = 4,
    parameter C_OUT = 4,
    parameter WL_BASE = 0
)(
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    input                    frame_start,
    output reg               out_valid,
    output reg signed [7:0]  pixel_out,
    output reg               out_last,

    input                    wl_we,
    input  [15:0]            wl_addr,
    input  signed [7:0]      wl_data,

    input  signed [31:0]     dw_m0,
    input  [4:0]             dw_shift,
    input  signed [31:0]     pw_m0,
    input  [4:0]             pw_shift
);

    localparam NUM_PX = IMG_W * IMG_H;

    localparam DW_KERNEL_OFFSET = 0;
    localparam DW_BIAS_OFFSET   = C_IN * 9;
    localparam DW_WEIGHT_DEPTH  = DW_BIAS_OFFSET + C_IN;
    localparam PW_KERNEL_OFFSET = DW_WEIGHT_DEPTH;
    localparam PW_BIAS_OFFSET   = PW_KERNEL_OFFSET + C_OUT * C_IN;
    localparam PW_WEIGHT_DEPTH  = PW_BIAS_OFFSET + C_OUT;

    // ---- weight registers (loaded over wl bus) ----
    reg signed [7:0]  dw_kernels [0:C_IN-1][0:8];
    reg signed [31:0] dw_biases [0:C_IN-1];
    reg signed [7:0]  pw_kernels [0:C_OUT-1][0:C_IN-1];
    reg signed [31:0] pw_biases [0:C_OUT-1];

    always @(posedge clk) begin
        if (wl_we && wl_addr >= WL_BASE + DW_KERNEL_OFFSET && wl_addr < WL_BASE + DW_KERNEL_OFFSET + C_IN*9)
            dw_kernels[(wl_addr - WL_BASE - DW_KERNEL_OFFSET)/9][(wl_addr - WL_BASE - DW_KERNEL_OFFSET)%9] <= wl_data;
        if (wl_we && wl_addr >= WL_BASE + DW_BIAS_OFFSET && wl_addr < WL_BASE + DW_WEIGHT_DEPTH)
            dw_biases[wl_addr - WL_BASE - DW_BIAS_OFFSET] <= {{24{wl_data[7]}}, wl_data};
        if (wl_we && wl_addr >= WL_BASE + PW_KERNEL_OFFSET && wl_addr < WL_BASE + PW_KERNEL_OFFSET + C_OUT*C_IN)
            pw_kernels[(wl_addr - WL_BASE - PW_KERNEL_OFFSET)/C_IN][(wl_addr - WL_BASE - PW_KERNEL_OFFSET)%C_IN] <= wl_data;
        if (wl_we && wl_addr >= WL_BASE + PW_BIAS_OFFSET && wl_addr < WL_BASE + PW_WEIGHT_DEPTH)
            pw_biases[wl_addr - WL_BASE - PW_BIAS_OFFSET] <= {{24{wl_data[7]}}, wl_data};
    end

    // ---- channel demux for input stream ----
    reg [$clog2(C_IN)-1:0] ch_cnt;
    always @(posedge clk) begin
        if (!rst_n || frame_start) ch_cnt <= 0;
        else if (in_valid) ch_cnt <= (ch_cnt == C_IN-1) ? 0 : ch_cnt + 1;
    end

    // ---- DW phase: C_IN parallel depthwise convs ----
    wire [C_IN-1:0] dw_out_valid;
    wire signed [7:0] dw_out [0:C_IN-1];

    genvar ch, t;
    generate
        for (ch = 0; ch < C_IN; ch = ch + 1) begin : dw_inst
            wire ch_valid = in_valid && (ch_cnt == ch);
            wire signed [71:0] kw_flat;
            for (t = 0; t < 9; t = t + 1) begin : kw_pack
                assign kw_flat[t*8 +: 8] = dw_kernels[ch][t];
            end
            dw_conv3x3_flat #(.IMG_W(IMG_W), .IMG_H(IMG_H))
            u_dw(
                .clk(clk), .rst_n(rst_n), .in_valid(ch_valid), .pixel_in(pixel_in),
                .frame_start(frame_start), .out_valid(dw_out_valid[ch]), .pixel_out(dw_out[ch]),
                .kw_flat(kw_flat), .bias(dw_biases[ch]), .rq_m0(dw_m0), .rq_shift(dw_shift)
            );
        end
    endgenerate

    // ---- DW output buffer ----
    reg signed [7:0] dw_buf [0:NUM_PX*C_IN-1];
    reg [$clog2(NUM_PX)-1:0] dw_wr_pos [0:C_IN-1];
    integer ci;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            for (ci = 0; ci < C_IN; ci = ci + 1) dw_wr_pos[ci] <= 0;
        end else begin
            for (ci = 0; ci < C_IN; ci = ci + 1) begin
                if (dw_out_valid[ci]) begin
                    dw_buf[dw_wr_pos[ci] * C_IN + ci] <= dw_out[ci];
                    dw_wr_pos[ci] <= dw_wr_pos[ci] + 1;
                end
            end
        end
    end

    reg dw_done;
    always @(posedge clk) begin
        if (!rst_n || frame_start) dw_done <= 1'b0;
        else if (dw_out_valid[C_IN-1] && dw_wr_pos[C_IN-1] == NUM_PX - 1) dw_done <= 1'b1;
    end

    // ---- pack PW weights (flat for pw_conv1x1_flat) ----
    wire signed [C_OUT*C_IN*8-1:0] pw_weight_flat;
    wire signed [C_OUT*32-1:0]     pw_bias_flat;
    genvar co, ci2;
    generate
        for (co = 0; co < C_OUT; co = co + 1) begin : pwpk
            for (ci2 = 0; ci2 < C_IN; ci2 = ci2 + 1) begin : pwpk_i
                assign pw_weight_flat[(co*C_IN + ci2)*8 +: 8] = pw_kernels[co][ci2];
            end
            assign pw_bias_flat[co*32 +: 32] = pw_biases[co];
        end
    endgenerate

    // ---- PW phase ----
    reg pw_in_valid;
    reg signed [7:0] pw_in_data;
    reg pw_in_last;
    wire pw_out_valid;
    wire signed [7:0] pw_out_data;
    wire pw_out_last;

    pw_conv1x1_flat #(.C_IN(C_IN), .C_OUT(C_OUT)) u_pw(
        .clk(clk), .rst_n(rst_n), .in_valid(pw_in_valid), .in_data(pw_in_data), .in_last(pw_in_last),
        .out_valid(pw_out_valid), .out_data(pw_out_data), .out_last(pw_out_last),
        .weight_flat(pw_weight_flat), .bias_flat(pw_bias_flat),
        .rq_m0(pw_m0), .rq_shift(pw_shift)
    );

    localparam PW_IDLE = 0, PW_FEED = 1, PW_WAIT = 2, PW_DONE = 3;
    reg [1:0] pw_state;
    reg [$clog2(NUM_PX)-1:0] pw_pos;
    reg [$clog2(C_IN)-1:0]   pw_feed_ci;
    reg [$clog2(C_OUT)-1:0]  pw_got_co;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            pw_state <= PW_IDLE; pw_pos <= 0; pw_feed_ci <= 0; pw_got_co <= 0;
            pw_in_valid <= 1'b0; pw_in_data <= 8'sd0; pw_in_last <= 1'b0;
            out_valid <= 1'b0; pixel_out <= 8'sd0; out_last <= 1'b0;
        end else begin
            pw_in_valid <= 1'b0; pw_in_last <= 1'b0;
            out_valid <= 1'b0; out_last <= 1'b0;
            if (pw_out_valid) begin
                out_valid <= 1'b1;
                pixel_out <= pw_out_data;
                out_last  <= pw_out_last;
            end
            case (pw_state)
                PW_IDLE: if (dw_done) begin pw_state <= PW_FEED; pw_pos <= 0; pw_feed_ci <= 0; end
                PW_FEED: begin
                    pw_in_valid <= 1'b1;
                    pw_in_data  <= dw_buf[pw_pos * C_IN + pw_feed_ci];
                    pw_in_last  <= (pw_feed_ci == C_IN - 1);
                    if (pw_feed_ci == C_IN - 1) begin
                        pw_feed_ci <= 0; pw_state <= PW_WAIT; pw_got_co <= 0;
                    end else pw_feed_ci <= pw_feed_ci + 1;
                end
                PW_WAIT: if (pw_out_valid) begin
                    if (pw_got_co == C_OUT - 1) begin
                        if (pw_pos == NUM_PX - 1) pw_state <= PW_DONE;
                        else begin pw_pos <= pw_pos + 1; pw_state <= PW_FEED; end
                    end else pw_got_co <= pw_got_co + 1;
                end
                PW_DONE: ;
                default: pw_state <= PW_IDLE;
            endcase
        end
    end

endmodule