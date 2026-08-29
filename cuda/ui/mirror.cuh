/* Purpose: Publish the cell grid and the live tables to the mirror a terminal reads.
 * Owns: The mirror addresses, the frame count and the device time the node takes.
 * Launch shape: One block of AOTX_MIRROR_THREADS; the node is the last of the raster graph.
 * Lifetime: From the bind of the mirror at the start to the close at the exit. */
#ifndef AOTX_UI_MIRROR_CUH
#define AOTX_UI_MIRROR_CUH

#include <stddef.h>

#include "seam/seam.cuh"
#include "ui/mirror.h"
#include "ui/ui.cuh"

/* Threads of the one block the node uses. One block is enough for 16 KB of cells at the
 * frame rate, and the block barrier orders the writes before the release store. */
#define AOTX_MIRROR_THREADS 256u

/* The grid goes over with 16-byte stores. The cells therefore start on a 16-byte boundary
 * of the snapshot and their byte count is a multiple of 16. */
typedef char aotx_mirror_check_cells[((offsetof(aotx_mirror_snapshot, cell) % 16u) == 0u
                                      && ((AOTX_MIRROR_CELLS * 2u) % 16u) == 0u) ? 1 : -1];

/* The mirror shows the grid the panel kernels write, so the two shapes are one shape. */
typedef char aotx_mirror_check_grid[(AOTX_MIRROR_COLS == AOTX_UI_COLS
                                     && AOTX_MIRROR_ROWS == AOTX_UI_ROWS
                                     && AOTX_MIRROR_PANELS == AOTX_UI_PANELS) ? 1 : -1];

/* What the node keeps between frames. A run with no mirror holds a null slot address and
 * the node writes nothing. */
typedef struct aotx_mirror_state {
    unsigned char *preamble;       /* device address of the mapped preamble */
    unsigned char *slot;           /* device address of the first snapshot */
    unsigned long long slot_bytes; /* bytes of one snapshot, a multiple of 16 */
    unsigned long long frame;      /* the sequence of the frame published last */
    unsigned long long tables[AOTX_MIRROR_SLOTS]; /* the frame each slot's tables hold */
    unsigned long long node_ns;    /* device time the node took over every frame */
} aotx_mirror_state;

extern __device__ aotx_mirror_state aotx_mirror;

/* Publish one snapshot with the rule of the tap ring and no consumer field. The node
 * release-stores zero into the sequence of the slot, writes the bytes, then release-stores
 * the frame. It takes the other slot at the frame after it. */
__global__ void aotx_ui_mirror(void);

/* Give the device the address of the mirror. A run whose rings hold no mirror leaves the
 * node inert. The return is zero when the node can publish. */
int aotx_mirror_bind(const aotx_seam_rings *rings);

/* Count the terminals the feeder says are attached. */
unsigned int aotx_mirror_attached(const aotx_seam_rings *rings);

/* What the thread of the mirror counted. */
typedef struct aotx_mirror_report {
    unsigned long long frames;   /* frames the node published, from every runner of it */
    unsigned long long launched; /* frames the thread of the mirror launched */
    unsigned long long node_ns;  /* device time the node took over the frames it published */
    unsigned long long idle;     /* turns the thread took with no terminal attached */
    unsigned int hz;             /* the frame rate of the last turn */
} aotx_mirror_report;

/* Build the raster graph and start the thread that runs it while a terminal is attached.
 * A run with a window calls neither of these. The thread of the window runs the graph, and
 * the mirror node is the last node of it. The return is zero when the thread runs. */
int aotx_mirror_start(const aotx_seam_rings *rings);

/* Stop the thread and give back the graph. */
void aotx_mirror_stop(void);

/* Read the counts of the thread and the device time of the node. */
void aotx_mirror_read(aotx_mirror_report *report);

/* Write one line with the frames and the microseconds of the node. */
void aotx_mirror_report_line(void);

#endif
