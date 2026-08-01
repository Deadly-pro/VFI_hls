// encoder_slice — Two-layer strided encoder (E1→E2) for Open-Frame-gen Nano.
// Chains two ds_conv_layer_integrated instances with stride-2 downsampling.
// Layer 1: IMG_W×IMG_H×C_IN  →  IMG_W/2×IMG_H/2×C_MID
// Layer 2: IMG_W/2×IMG_H/2×C_MID  →  IMG_W/4×IMG_H/4×C_OUT

module encoder_slice #(
    parameter IMG_W      = 8,
    parameter IMG_H      = 8,
    parameter C_IN       = 6,
    parameter C_MID      = 48,
    parameter C_OUT      = 96,
    parameter DW_MEM_DEPTH = 1024,
    parameter DW_MEM_AW    = 10,
    parameter PW_MEM_DEPTH = 2048,
    parameter PW_MEM_AW    = 11
)(
    input                    clk,
    input                    rst_n,
    input                    in_valid,
    input  signed [7:0]      pixel_in,
    input                    frame_start,
    output reg               out_valid,
    output reg signed [7:0]  pixel_out,
    output reg               out_last,

    // Shared weight-load bus
    input                    wl_we,
    input  [DW_MEM_AW-1:0]   wl_addr,
    input  signed [7:0]      wl_data,

    // Requant params per layer
    input  signed [31:0]     e1_dw_m0,
    input  [4:0]             e1_dw_shift,
    input  signed [31:0]     e1_pw_m0,
    input  [4:0]             e1_pw_shift,
    input  signed [31:0]     e2_dw_m0,
    input  [4:0]             e2_dw_shift,
    input  signed [31:0]     e2_pw_m0,
    input  [4:0]             e2_pw_shift
);

    // ---- Layer 1 (E1): stride-2 downsampling 8×8×C_IN → 4×4×C_MID ----
    wire e1_out_valid;
    wire signed [7:0] e1_pixel_out;
    wire e1_out_last;

    // E1 weight-load uses same bus, separate address ranges
    // E1 DW: [0 .. C_IN*9+C_IN-1], E1 PW: [0 .. C_MID*C_IN+C_MID-1]

    ds_conv_layer_integrated #(
        .IMG_W(IMG_W),
        .IMG_H(IMG_H),
        .C_IN(C_IN),
        .C_OUT(C_MID),
        .DW_MEM_DEPTH(DW_MEM_DEPTH),
        .DW_MEM_AW(DW_MEM_AW),
        .PW_MEM_DEPTH(PW_MEM_DEPTH),
        .PW_MEM_AW(PW_MEM_AW)
    ) u_e1 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .pixel_in(pixel_in),
        .frame_start(frame_start),
        .out_valid(e1_out_valid),
        .pixel_out(e1_pixel_out),
        .out_last(e1_out_last),
        .wl_we(wl_we),
        .wl_addr(wl_addr),
        .wl_data(wl_data),
        .dw_m0(e1_dw_m0),
        .dw_shift(e1_dw_shift),
        .pw_m0(e1_pw_m0),
        .pw_shift(e1_pw_shift)
    );

    // ---- Stride-2 filter: pass only even (row,col) positions from E1 ----
    // E1 outputs 8×8×C_MID = 64*C_MID pixels in channel-interleaved order
    // We need to keep only positions where (row%2==0 && col%2==0) → 4×4 = 16 positions
    localparam E1_NUM_PX = IMG_W * IMG_H;           // 64
    localparam E1_OUT_PX = (IMG_W/2) * (IMG_H/2);   // 16
    localparam E2_NUM_PX = E1_OUT_PX;               // 16
    localparam STRIDE = 2;

    reg [$clog2(E1_NUM_PX):0] e1_pos_cnt;  // counts E1 output positions
    reg [$clog2(C_MID)-1:0]   e1_ch_cnt;   // channel counter
    reg                       e1_stride_pass;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            e1_pos_cnt    <= 0;
            e1_ch_cnt     <= 0;
            e1_stride_pass <= 1'b0;
        end else if (e1_out_valid) begin
            // Check if this position passes stride-2 filter
            integer row = e1_pos_cnt / IMG_W;
            integer col = e1_pos_cnt % IMG_W;
            e1_stride_pass <= (row % STRIDE == 0) && (col % STRIDE == 0);

            if (e1_ch_cnt == C_MID - 1) begin
                e1_ch_cnt <= 0;
                e1_pos_cnt <= e1_pos_cnt + 1;
            end else begin
                e1_ch_cnt <= e1_ch_cnt + 1;
            end
        end
    end

    // ---- Layer 2 (E2): stride-2 downsampling 4×4×C_MID → 2×2×C_OUT ----
    // E2 takes filtered E1 outputs as input
    wire e2_in_valid = e1_out_valid && e1_stride_pass;
    wire e2_frame_start = frame_start;  // sync with original frame

    // E2 weight-load: separate address space (offset after E1 weights)
    // In practice, would use separate wl_addr ranges or separate bus
    // For now, share bus - user must load E1 then E2 weights sequentially

    ds_conv_layer_integrated #(
        .IMG_W(IMG_W/2),
        .IMG_H(IMG_H/2),
        .C_IN(C_MID),
        .C_OUT(C_OUT),
        .DW_MEM_DEPTH(DW_MEM_DEPTH),
        .DW_MEM_AW(DW_MEM_AW),
        .PW_MEM_DEPTH(PW_MEM_DEPTH),
        .PW_MEM_AW(PW_MEM_AW)
    ) u_e2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(e2_in_valid),
        .pixel_in(e1_pixel_out),
        .frame_start(e2_frame_start),
        .out_valid(out_valid),
        .pixel_out(pixel_out),
        .out_last(out_last),
        .wl_we(wl_we),
        .wl_addr(wl_addr),
        .wl_data(wl_data),
        .dw_m0(e2_dw_m0),
        .dw_shift(e2_dw_shift),
        .pw_m0(e2_pw_m0),
        .pw_shift(e2_pw_shift)
    );

endmodule