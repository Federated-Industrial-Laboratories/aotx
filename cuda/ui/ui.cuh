/* Purpose: Compose the text grid and rasterize it.
 * Owns: The cell grid, the panel table, the font and the pixel buffer.
 * Launch shape: One block for each panel; one thread for each pixel.
 * Lifetime: The whole run. */
#ifndef AOTX_UI_CUH
#define AOTX_UI_CUH

#include "cli/cli.cuh"

/* The grid is fixed for this version: 160 columns of 50 rows, each cell 8 by 16 pixels. */
#define AOTX_UI_COLS          160u
#define AOTX_UI_ROWS          50u
#define AOTX_UI_CELLS         (AOTX_UI_COLS * AOTX_UI_ROWS)
#define AOTX_UI_CELL_WIDTH    8u
#define AOTX_UI_CELL_HEIGHT   16u
#define AOTX_UI_WIDTH         (AOTX_UI_COLS * AOTX_UI_CELL_WIDTH)
#define AOTX_UI_HEIGHT        (AOTX_UI_ROWS * AOTX_UI_CELL_HEIGHT)
#define AOTX_UI_PIXELS        (AOTX_UI_WIDTH * AOTX_UI_HEIGHT)
#define AOTX_UI_PIXEL_BYTES   (AOTX_UI_PIXELS * 4u)

/* The font holds the code points 32 to 126 and one replacement box after them. */
#define AOTX_UI_GLYPH_FIRST   AOTX_CLI_CODE_FIRST
#define AOTX_UI_GLYPH_LAST    AOTX_CLI_CODE_LAST
#define AOTX_UI_GLYPH_ROWS    16u
#define AOTX_UI_GLYPHS        96u
#define AOTX_UI_GLYPH_BOX     95u
#define AOTX_UI_GLYPH_SPACE   0u

/* Three attributes give three fixed colors. A cell carries no other decoration. */
#define AOTX_UI_NORMAL        0u
#define AOTX_UI_DIM           1u
#define AOTX_UI_HIGH          2u
#define AOTX_UI_ATTRS         3u

/* The pixel word holds red in the low byte, so the buffer is RGBA in memory order. */
#define AOTX_UI_RGBA(r, g, b) (0xff000000u | ((unsigned int)(b) << 16) \
                               | ((unsigned int)(g) << 8) | (unsigned int)(r))
#define AOTX_UI_BACK          AOTX_UI_RGBA(11u, 13u, 16u)
#define AOTX_UI_COLOR_NORMAL  AOTX_UI_RGBA(200u, 205u, 210u)
#define AOTX_UI_COLOR_DIM     AOTX_UI_RGBA(110u, 115u, 120u)
#define AOTX_UI_COLOR_HIGH    AOTX_UI_RGBA(255u, 214u, 102u)

/* Panel identities. The table below gives the rectangle of each one. */
#define AOTX_UI_CONSOLE       0u
#define AOTX_UI_AGENTS        1u
#define AOTX_UI_BUS           2u
#define AOTX_UI_ARENA         3u
#define AOTX_UI_TICK          4u
#define AOTX_UI_SEAM          5u
#define AOTX_UI_PANELS        6u

/* Threads of one panel block. The block covers the rows of the panel and its cells. */
#define AOTX_UI_PANEL_THREADS 128u

/* Rows of records a panel holds. The console shows this many lines above the command line,
 * and the bus panel shows this many messages at the most. */
#define AOTX_UI_BUS_MAX       32u
#define AOTX_UI_LINE_MAX      32u
#define AOTX_UI_BUS_KINDS     0xfeu

/* The raster grid. Each thread takes one pixel of each stride over the buffer. */
#define AOTX_UI_RASTER_BLOCKS  1024u
#define AOTX_UI_RASTER_THREADS 256u

typedef struct aotx_ui_cell {
    unsigned char glyph;   /* index into the font array */
    unsigned char attr;    /* AOTX_UI_NORMAL, AOTX_UI_DIM or AOTX_UI_HIGH */
} aotx_ui_cell;

typedef struct aotx_ui_panel {
    unsigned short col;    /* first column of the panel */
    unsigned short row;    /* first row of the panel */
    unsigned short cols;   /* columns the panel holds */
    unsigned short rows;   /* rows the panel holds, the title row included */
} aotx_ui_panel;

extern __constant__ aotx_ui_panel aotx_ui_panel_table[AOTX_UI_PANELS];
extern __constant__ unsigned char aotx_ui_font[AOTX_UI_GLYPHS][AOTX_UI_GLYPH_ROWS];
extern __device__ aotx_ui_cell aotx_ui_grid[AOTX_UI_CELLS];
extern __device__ unsigned int aotx_ui_pixel[AOTX_UI_PIXELS];

/* Give the glyph of a code point. A code point outside the font gives the box. */
__device__ __forceinline__ unsigned char aotx_ui_glyph(unsigned int code)
{
    if (code < AOTX_UI_GLYPH_FIRST || code > AOTX_UI_GLYPH_LAST) {
        return (unsigned char)AOTX_UI_GLYPH_BOX;
    }
    return (unsigned char)(code - AOTX_UI_GLYPH_FIRST);
}

/* Put one cell of a panel. A position outside the panel is dropped. */
__device__ __forceinline__ void aotx_ui_put(const aotx_ui_panel *panel, unsigned int row,
                                            unsigned int col, unsigned char glyph,
                                            unsigned int attr)
{
    if (row >= panel->rows || col >= panel->cols) {
        return;
    }
    unsigned int at = ((unsigned int)panel->row + row) * AOTX_UI_COLS
                    + (unsigned int)panel->col + col;
    aotx_ui_grid[at].glyph = glyph;
    aotx_ui_grid[at].attr = (unsigned char)attr;
}

/* Blank every cell of a panel. Each thread of the block takes a share of the cells. */
__device__ __forceinline__ void aotx_ui_blank(const aotx_ui_panel *panel)
{
    unsigned int cells = (unsigned int)panel->rows * (unsigned int)panel->cols;
    for (unsigned int i = threadIdx.x; i < cells; i += blockDim.x) {
        aotx_ui_put(panel, i / panel->cols, i % panel->cols,
                    (unsigned char)AOTX_UI_GLYPH_SPACE, AOTX_UI_DIM);
    }
}

/* Blank one row of a panel. A row whose record went out of the ring while it was built
 * shows nothing rather than a mix of two records. */
__device__ __forceinline__ void aotx_ui_blank_row(const aotx_ui_panel *panel,
                                                  unsigned int row)
{
    for (unsigned int col = 0u; col < panel->cols; ++col) {
        aotx_ui_put(panel, row, col, (unsigned char)AOTX_UI_GLYPH_SPACE, AOTX_UI_DIM);
    }
}

/* Put a run of bytes on one row of a panel. Returns the column after the last byte. */
__device__ __forceinline__ unsigned int aotx_ui_text(const aotx_ui_panel *panel,
                                                     unsigned int row, unsigned int col,
                                                     const char *text, unsigned int length,
                                                     unsigned int attr)
{
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_ui_put(panel, row, col + i, aotx_ui_glyph((unsigned char)text[i]), attr);
    }
    return col + length;
}

/* Put a text that ends with a zero byte. */
__device__ __forceinline__ unsigned int aotx_ui_say(const aotx_ui_panel *panel,
                                                    unsigned int row, unsigned int col,
                                                    const char *text, unsigned int attr)
{
    return aotx_ui_text(panel, row, col, text, aotx_cli_length(text), attr);
}

/* Put a decimal value. Returns the column after the last digit. */
__device__ __forceinline__ unsigned int aotx_ui_number(const aotx_ui_panel *panel,
                                                       unsigned int row, unsigned int col,
                                                       unsigned long long value,
                                                       unsigned int attr)
{
    char digits[24];
    unsigned int count = aotx_cli_utoa(value, digits, (unsigned int)sizeof digits);
    return aotx_ui_text(panel, row, col, digits, count, attr);
}

/* Put a name, then a value, on one row: the name dim and the value normal. */
__device__ __forceinline__ void aotx_ui_field(const aotx_ui_panel *panel, unsigned int row,
                                              const char *name, unsigned long long value)
{
    unsigned int col = aotx_ui_say(panel, row, 1u, name, AOTX_UI_DIM);
    aotx_ui_number(panel, row, col + 1u, value, AOTX_UI_NORMAL);
}

/* Put the body bytes of a record on one row. The sequence is read again after the copy, so
 * a record that the ring wrote over while the row was built shows nothing. */
__device__ __forceinline__ void aotx_ui_body(const aotx_ui_panel *panel, unsigned int row,
                                             unsigned int col, unsigned long long seq,
                                             unsigned int type, unsigned int attr)
{
    if (seq == 0ull) {
        return;
    }
    const volatile aotx_record_header *header = aotx_cli_slot(seq);
    if (!aotx_cli_holds(header, seq, type)) {
        return;
    }
    unsigned int length = header->body_len;
    if (length > AOTX_BODY_BYTES) {
        length = AOTX_BODY_BYTES;
    }
    if (col + length > panel->cols) {
        length = (col < panel->cols) ? (panel->cols - col) : 0u;
    }
    const volatile unsigned char *body = (const volatile unsigned char *)header
                                       + AOTX_HEADER_BYTES;
    for (unsigned int i = 0u; i < length; ++i) {
        aotx_ui_put(panel, row, col + i, aotx_ui_glyph(body[i]), attr);
    }
    if (!aotx_cli_holds(header, seq, type)) {
        for (unsigned int i = 0u; i < length; ++i) {
            aotx_ui_put(panel, row, col + i, (unsigned char)AOTX_UI_GLYPH_SPACE, AOTX_UI_DIM);
        }
    }
}

/* Fill seqs with the newest records of a type, the newest first, at most max of them. The
 * return is the count filled. Every thread of the block calls this. */

/* The block walks the ring one chunk at a time, so the loads of a chunk are in flight
 * together. A walk by one thread for each row costs milliseconds and holds the frame. */
__device__ __forceinline__ unsigned int aotx_ui_recent(unsigned int type, unsigned int max,
                                                       unsigned long long *seqs)
{
    __shared__ unsigned int aotx_ui_warp_count[AOTX_UI_PANEL_THREADS / 32u];
    __shared__ unsigned int aotx_ui_warp_base[AOTX_UI_PANEL_THREADS / 32u];
    __shared__ unsigned int aotx_ui_found;
    __shared__ unsigned int aotx_ui_bound;

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warps = blockDim.x >> 5;
    unsigned long long tail = aotx_seam.dev.tail;
    unsigned long long window = aotx_seam.dev.slot_count;
    if (window > tail) {
        window = tail;
    }
    if (threadIdx.x == 0u) {
        aotx_ui_found = 0u;
        aotx_ui_bound = 0xffffffffu;
    }
    __syncthreads();

    for (unsigned long long base = 0ull; base < window; base += (unsigned long long)blockDim.x) {
        unsigned long long at = base + (unsigned long long)threadIdx.x;
        unsigned long long seq = 0ull;
        unsigned int hit = 0u;
        if (at < window && at < (unsigned long long)aotx_ui_bound) {
            seq = tail - at;
            const volatile aotx_record_header *header = aotx_cli_slot(seq);
            unsigned long long got = header->seq;
            if (got != seq) {
                /* A slot that a newer record took ends the run of records that are left. */
                if (got != 0ull) {
                    atomicMin(&aotx_ui_bound, (unsigned int)at);
                }
            } else if (header->magic == AOTX_WIRE_MAGIC
                       && header->type == (unsigned char)type) {
                hit = 1u;
            }
        }
        __syncthreads();
        /* A slot after the end of the run does not count, whatever it holds. */
        if (at >= (unsigned long long)aotx_ui_bound) {
            hit = 0u;
        }
        unsigned int mask = __ballot_sync(0xffffffffu, hit != 0u);
        if (lane == 0u) {
            aotx_ui_warp_count[warp] = (unsigned int)__popc(mask);
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            unsigned int run = 0u;
            for (unsigned int w = 0u; w < warps; ++w) {
                aotx_ui_warp_base[w] = run;
                run += aotx_ui_warp_count[w];
            }
            aotx_ui_warp_count[0] = run;
        }
        __syncthreads();
        unsigned int mine = aotx_ui_warp_base[warp]
                          + (unsigned int)__popc(mask & ((1u << lane) - 1u));
        if (hit != 0u && aotx_ui_found + mine < max) {
            seqs[aotx_ui_found + mine] = seq;
        }
        unsigned int run = aotx_ui_warp_count[0];
        __syncthreads();
        if (threadIdx.x == 0u) {
            aotx_ui_found += run;
        }
        __syncthreads();
        if (aotx_ui_found >= max || aotx_ui_bound != 0xffffffffu) {
            break;
        }
    }
    return (aotx_ui_found < max) ? aotx_ui_found : max;
}

__global__ void aotx_ui_console(void);
__global__ void aotx_ui_agents(void);
__global__ void aotx_ui_bus(void);
__global__ void aotx_ui_arena(void);
__global__ void aotx_ui_tick(void);
__global__ void aotx_ui_seam(void);
__global__ void aotx_ui_raster(void);

/* What the host glue keeps to run the raster graph. The graph holds the six panel kernels
 * and the raster kernel, and its shape never changes. */
typedef struct aotx_ui_graph {
    cudaStream_t stream;    /* the highest priority stream of this device */
    cudaEvent_t event;      /* the frame waits on this event */
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    int priority;           /* the priority value the stream was made with */
} aotx_ui_graph;

/* Capture the panel kernels and the raster kernel once, on a stream of the highest
 * priority. The return is zero when the graph is ready. */
int aotx_ui_graph_build(aotx_ui_graph *graph);

/* Launch one frame and wait for it on the event. */
void aotx_ui_graph_run(aotx_ui_graph *graph);

/* Give the device address of the pixel buffer, for the copy into the pixel buffer object. */
void *aotx_ui_graph_pixels(void);

/* Give back the graph, the stream and the event. */
void aotx_ui_graph_close(aotx_ui_graph *graph);

/* Make the window and its drawing context. This runs before the first driver call, so the
 * context of the system binds to the device that drives the display. */
int aotx_ui_window_open(void);

/* Make the texture and the two pixel buffer objects, and build the raster graph. The key
 * events go to the descriptor, one 16-byte frame for each event. */
int aotx_ui_window_bind(int keys_fd);

/* Draw one frame at the refresh of the display. The return is zero when the window closes. */
int aotx_ui_window_frame(void);

/* The bound above which a frame is late. A display at 60 Hz gives a frame every 16.7 ms. */
#define AOTX_UI_LATE_NS  20000000ull

/* What the window counted while it drew. An interval is the time from the end of one frame
 * to the end of the next. */
typedef struct aotx_ui_frame_report {
    unsigned long long frames;     /* frames the window drew */
    unsigned long long mean_ns;    /* mean interval between two frames */
    unsigned long long worst_ns;   /* longest interval between two frames */
    unsigned long long late;       /* intervals above the bound */
    unsigned long long dropped;    /* key events that the pipe could not take */
} aotx_ui_frame_report;

/* Give the counts the window kept. */
void aotx_ui_window_report(aotx_ui_frame_report *report);

/* Give the priority value of the stream the raster graph runs on. */
int aotx_ui_window_priority(void);

/* Give back the texture, the buffers, the graph and the window. */
void aotx_ui_window_close(void);

#endif
