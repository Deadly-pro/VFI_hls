// vfi_synth — VFI accelerator top: warp(t,flow), warp(t+1,flow), mask-blend.
//
// End-to-end frame interpolation core (the warp+blend tail of nano_v16.onnx):
//   out = mask * warp(t, flow) + (1-mask) * warp(t+1, flow)
//
// Multi-channel (RGB): NCH independent channel paths, flow/mask shared.
//
// Streaming contract (throughput 1/NCH):
//   Capture: channel-interleaved (ch0_pos0, ch1_pos0, ..., ch0_pos1, ...) on
//            capture_valid for t and t+1 in parallel.
//   Warp+blend: one (flow_x, flow_y, mask) per spatial position; the position
//            is consumed once per NCH cycles (output serializes over NCH
//            cycles). out_valid pulses NCH times per position, channel-major.
//
// Flow is displacement in pixels, fixed-point Q<FLOW_Q> (see warp_unit).

module vfi_synth #(
    parameter IMG_W  = 16,
    parameter IMG_H  = 16,
    parameter FLOW_Q = 8,
    parameter FLOW_W = 16,
    parameter NCH    = 3
)(
    input                    clk,
    input                    rst_n,
    // frame capture: t and t+1 arrive together, channel-interleaved
    input                    capture_valid,
    input  signed [7:0]      pixel_t_in,
    input  signed [7:0]      pixel_t1_in,
    // flow + mask stream (one position per NCH cycles, row-major)
    input                    wb_valid,
    input  signed [FLOW_W-1:0] flow_x,
    input  signed [FLOW_W-1:0] flow_y,
    input  [7:0]             mask,
    // output (channel-major per position)
    output reg               out_valid,
    output reg signed [7:0]  pixel_out
);

    localparam NUM_PX = IMG_W * IMG_H;

    // capture demux: route pixels to per-channel warp capture
    reg [$clog2(NCH)-1:0] ch_cnt;

    wire [NCH-1:0] w_valid_t, w_valid_t1;
    wire signed [7:0] w_t  [0:NCH-1];
    wire signed [7:0] w_t1 [0:NCH-1];

    genvar g;
    generate
        for (g = 0; g < NCH; g = g + 1) begin : ch
            wire g_valid = capture_valid && (ch_cnt == g);

            warp_unit #(
                .IMG_W(IMG_W), .IMG_H(IMG_H), .FLOW_Q(FLOW_Q), .FLOW_W(FLOW_W)
            ) u_warp_t (
                .clk(clk), .rst_n(rst_n),
                .in_valid(g_valid), .pixel_in(pixel_t_in),
                .flow_valid(wb_valid), .flow_x(flow_x), .flow_y(flow_y),
                .out_valid(w_valid_t[g]), .pixel_out(w_t[g])
            );

            warp_unit #(
                .IMG_W(IMG_W), .IMG_H(IMG_H), .FLOW_Q(FLOW_Q), .FLOW_W(FLOW_W)
            ) u_warp_t1 (
                .clk(clk), .rst_n(rst_n),
                .in_valid(g_valid), .pixel_in(pixel_t1_in),
                .flow_valid(wb_valid), .flow_x(flow_x), .flow_y(flow_y),
                .out_valid(w_valid_t1[g]), .pixel_out(w_t1[g])
            );
        end
    endgenerate

    // per-channel blend (both warps emit on same cycle, so w_valid is shared)
    wire [NCH-1:0] b_valid;
    wire signed [7:0] b_out [0:NCH-1];

    genvar h;
    generate
        for (h = 0; h < NCH; h = h + 1) begin : blend_gen
            blend_unit u_blend (
                .clk(clk), .rst_n(rst_n),
                .in_valid(w_valid_t[h]), .in_a(w_t[h]), .in_b(w_t1[h]), .mask(mask),
                .out_valid(b_valid[h]), .pixel_out(b_out[h])
            );
        end
    endgenerate

    // output serialization: when a warp+blend position completes (blend valid),
    // stream NCH channels out over NCH cycles
    reg [$clog2(NCH):0] emit_cnt;   // 0 = idle, 1..NCH = emitting

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ch_cnt   <= 0;
            emit_cnt <= 0;
            out_valid <= 1'b0;
            pixel_out <= 8'sd0;
        end else begin
            if (capture_valid) begin
                ch_cnt <= (ch_cnt == NCH - 1) ? 0 : ch_cnt + 1;
            end

            out_valid <= 1'b0;
            if (b_valid[0]) begin
                // this position's NCH blends are all ready; start emitting
                emit_cnt  <= 1;
                out_valid <= 1'b1;
                pixel_out <= b_out[0];
            end else if (emit_cnt != 0) begin
                if (emit_cnt == NCH) begin
                    emit_cnt <= 0;              // done this position
                end else begin
                    out_valid <= 1'b1;
                    pixel_out <= b_out[emit_cnt];
                    emit_cnt  <= emit_cnt + 1;
                end
            end
        end
    end

endmodule