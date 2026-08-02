`timescale 1ns/1ps
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
    input  [15:0]            wl_addr,
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

    // E1 weight-load uses same bus, distinct address window [0, E1_WEIGHT_SPACE)
    localparam E1_DW_SPACE  = C_IN * 9 + C_IN;              // DW kernels + biases
    localparam E1_PW_SPACE  = C_MID * C_IN + C_MID;         // PW kernels + biases
    localparam E1_WEIGHT_SPACE = E1_DW_SPACE + E1_PW_SPACE;
    // E2 loads at [E1_WEIGHT_SPACE, E1_WEIGHT_SPACE + E2_WEIGHT_SPACE)
    localparam E2_DW_SPACE  = C_MID * 9 + C_MID;
    localparam E2_PW_SPACE  = C_OUT * C_MID + C_OUT;
    localparam E2_WEIGHT_SPACE = E2_DW_SPACE + E2_PW_SPACE;

    ds_conv_layer_integrated #(
        .IMG_W(IMG_W),
        .IMG_H(IMG_H),
        .C_IN(C_IN),
        .C_OUT(C_MID),
        .DW_MEM_DEPTH(DW_MEM_DEPTH),
        .DW_MEM_AW(DW_MEM_AW),
        .PW_MEM_DEPTH(PW_MEM_DEPTH),
        .PW_MEM_AW(PW_MEM_AW),
        .WL_BASE(0)
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

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            e1_pos_cnt    <= 0;
            e1_ch_cnt     <= 0;
        end else if (e1_out_valid) begin
            if (e1_ch_cnt == C_MID - 1) begin
                e1_ch_cnt <= 0;
                e1_pos_cnt <= e1_pos_cnt + 1;
            end else begin
                e1_ch_cnt <= e1_ch_cnt + 1;
            end
        end
    end

    // ---- Layer 2 (E2): stride-2 downsampling 4×4×C_MID → 2×2×C_OUT ----
    // E2 takes filtered E1 outputs as input. Keep only even (row,col)
    // positions of E1 — combinational on the current position (e1_pos_cnt
    // is stable across the C_MID channel cycles of one position).
    wire e1_pass_cur =
        ((e1_pos_cnt / IMG_W) % STRIDE == 0) && ((e1_pos_cnt % IMG_W) % STRIDE == 0);
    wire e2_in_valid = e1_out_valid && e1_pass_cur;
    wire e2_frame_start = frame_start;  // sync with original frame

    // E2 weight-load: separate address space (offset after E1 weights)
    // In practice, would use separate wl_addr ranges or separate bus
    // For now, share bus - user must load E1 then E2 weights sequentially

    wire e2_out_valid;
    wire signed [7:0] e2_pixel_out;
    wire e2_out_last;

    ds_conv_layer_integrated #(
        .IMG_W(IMG_W/2),
        .IMG_H(IMG_H/2),
        .C_IN(C_MID),
        .C_OUT(C_OUT),
        .DW_MEM_DEPTH(DW_MEM_DEPTH),
        .DW_MEM_AW(DW_MEM_AW),
        .PW_MEM_DEPTH(PW_MEM_DEPTH),
        .PW_MEM_AW(PW_MEM_AW),
        .WL_BASE(E1_WEIGHT_SPACE)
    ) u_e2 (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(e2_in_valid),
        .pixel_in(e1_pixel_out),
        .frame_start(e2_frame_start),
        .out_valid(e2_out_valid),
        .pixel_out(e2_pixel_out),
        .out_last(e2_out_last),
        .wl_we(wl_we),
        .wl_addr(wl_addr),
        .wl_data(wl_data),
        .dw_m0(e2_dw_m0),
        .dw_shift(e2_dw_shift),
        .pw_m0(e2_pw_m0),
        .pw_shift(e2_pw_shift)
    );

    // ---- Stride-2 filter on E2's output: keep even (row,col) positions ----
    // E2 emits its full 4×4×C_OUT; downsample to 2×2×C_OUT.
    localparam E2_OUT_W = IMG_W / 2;   // 4
    localparam E2_OUT_H = IMG_H / 2;   // 4
    reg [$clog2(E2_NUM_PX):0] e2_pos_cnt;
    reg [$clog2(C_OUT)-1:0]   e2_ch_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            e2_pos_cnt <= 0;
            e2_ch_cnt  <= 0;
        end else if (e2_out_valid) begin
            if (e2_ch_cnt == C_OUT - 1) begin
                e2_ch_cnt <= 0;
                e2_pos_cnt <= e2_pos_cnt + 1;
            end else begin
                e2_ch_cnt <= e2_ch_cnt + 1;
            end
        end
    end

    wire e2_pass =
        ((e2_pos_cnt / E2_OUT_W) % STRIDE == 0) && ((e2_pos_cnt % E2_OUT_W) % STRIDE == 0);
    wire e2_pass_last = e2_out_valid && e2_pass && (e2_ch_cnt == C_OUT - 1);

    reg signed [7:0] pixel_out_r;
    reg              out_valid_r;
    reg              out_last_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n || frame_start) begin
            out_valid_r <= 1'b0;
            pixel_out_r <= 8'sd0;
            out_last_r  <= 1'b0;
        end else begin
            out_valid_r <= e2_out_valid && e2_pass;
            if (e2_out_valid && e2_pass) begin
                pixel_out_r <= e2_pixel_out;
                out_last_r  <= (e2_pos_cnt == E2_NUM_PX - 1) && (e2_ch_cnt == C_OUT - 1);
            end
        end
    end

    assign out_valid = out_valid_r;
    assign pixel_out = pixel_out_r;
    assign out_last  = out_last_r;

endmodule