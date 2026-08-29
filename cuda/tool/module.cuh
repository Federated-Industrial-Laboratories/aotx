/* Purpose: Hold the batch and the output of the device tool modules of the tick graph.
 * Owns: The rows, the scratch and the output the module nodes read and write.
 * Launch shape: Device state; the fill writes it, a module node reads it, the step reaps it.
 * Lifetime: The whole run.
 *
 * A device tool is a module the driver loads. It runs as one node of the tick graph with
 * two parameters: the address of its batch and the address of the output. The take flag of
 * a row says whether the row is a request for that module in this tick. Each node
 * therefore holds rows of its own. One request belongs to one tool, so the scratch and the
 * output are the same tables for every node. */
#ifndef AOTX_TOOL_MODULE_CUH
#define AOTX_TOOL_MODULE_CUH

#include "aotx_tool.h"

#include "catalog/catalog.cuh"
#include "tool/tool.cuh"

/* Threads of one block of a module node. The node runs one block for each request row of
 * the profile, and the row of a block is blockIdx.x. */
#define AOTX_TOOL_MODULE_THREADS 256u

/* Bytes of the directory of the module directories of a run. The loader takes it from the
 * boot, and it opens a module below it when the path of the import does not open. */
#define AOTX_MODULE_ROOT_BYTES   512u

/* Bytes of the module file name and of the kernel name the manifest gives. */
#define AOTX_TOOL_MODULE_FILE    64u

/* The value that stands for no node of the tick graph. */
#define AOTX_TOOL_MODULE_NONE    AOTX_TOOL_MODULES

/* The tables the module nodes read and write. The batch of a node points into them, and a
 * kernel sets those pointers, so no host address enters the device state. */
typedef struct aotx_tool_module_state {
    aotx_tool_batch      batch[AOTX_TOOL_MODULES];
    aotx_tool_output     out;
    aotx_tool_row        row[AOTX_TOOL_MODULES * AOTX_SLOTS];
    aotx_tool_output_row head[AOTX_SLOTS];
    char                 text[AOTX_SLOTS * AOTX_TOOL_RESULT_BYTES];
    unsigned char        scratch[AOTX_SLOTS * AOTX_TOOL_SCRATCH_BYTES];
    unsigned int         entry[AOTX_TOOL_MODULES];  /* the catalog entry of each node */
    unsigned int         nodes;    /* nodes the tick graph holds */
    unsigned int         gen;      /* the catalog number the nodes were built with */
    unsigned int         took;     /* rows the fill gave the modules */
    unsigned int         gave;     /* rows a module answered */
    unsigned int         over;     /* rows whose length went past the bound */
    unsigned int         untaken;  /* rows a module wrote that no fill gave it */
    unsigned int         bare;     /* calls to a device tool that holds no node */
    unsigned int         captures; /* captures the pump made after the first one */
    unsigned int         before;   /* nodes of the tick graph before the last capture */
    unsigned int         after;    /* nodes of the tick graph after it */
    unsigned int         took_us;  /* microseconds the last capture took */
} aotx_tool_module_state;

extern __device__ aotx_tool_module_state aotx_tool_modules;

/* One row of the plan the host glue reads: what to open, what to check and what to find. */
typedef struct aotx_tool_module_row {
    unsigned int  entry;      /* the catalog entry of the tool */
    unsigned int  import;     /* the number of the import that installed it */
    char          name[AOTX_CATALOG_NAME_BYTES];
    char          path[AOTX_IMPORT_PATH_BYTES];   /* the directory of the import */
    char          file[AOTX_TOOL_MODULE_FILE];    /* the module file below it */
    char          kernel[AOTX_CATALOG_NAME_BYTES];/* the kernel of the module */
    unsigned char digest[32];
} aotx_tool_module_row;

/* The plan of one capture. A kernel builds it from the catalog and the host glue reads it
 * back with one copy, so no host code parses a manifest. */
typedef struct aotx_tool_module_plan {
    aotx_tool_module_row row[AOTX_TOOL_MODULES];
    unsigned int         rows;
    unsigned int         gen;   /* the catalog number this plan was built from */
} aotx_tool_module_plan;

extern __device__ aotx_tool_module_plan aotx_tool_module_list;

/* Build the plan of the device tools the catalog holds. One thread makes the call. */
__global__ void aotx_tool_module_scan(void);

/* Take the plan the host glue loaded: the entry of each node, the pointers of each batch
 * and the pointers of the output. One thread makes the call. */
__global__ void aotx_tool_module_bind(unsigned int nodes);

/* Write the record of one capture of the tick graph: the tick, the node count before and
 * after, and the microseconds it took. One thread makes the call. */
__global__ void aotx_tool_module_note(unsigned int before, unsigned int after,
                                      unsigned int took_us);

/* Give one entry the state REFUSED with a reason, and write the console line and the bus
 * note that name it. The host glue launches this for a module file it refuses. A file does
 * not open, or its digest differs, or it holds no kernel of that name. */
__global__ void aotx_tool_module_refuse(unsigned int entry, unsigned int why);

/* The node of a catalog entry, or AOTX_TOOL_MODULE_NONE. */
__device__ __forceinline__ unsigned int aotx_tool_module_node(unsigned int entry)
{
    for (unsigned int m = 0u; m < aotx_tool_modules.nodes; ++m) {
        if (aotx_tool_modules.entry[m] == entry) {
            return m;
        }
    }
    return (unsigned int)AOTX_TOOL_MODULE_NONE;
}

/* Fill the rows of every module node from the request table. The fill step of the tool
 * path calls this with one thread for each request slot. */
__device__ void aotx_tool_module_fill(unsigned int slot);

/* Take the answer of a module for one request. The return is 1 when the row is done and
 * the result stands in the request. The tool step calls this for its slot. */
__device__ int aotx_tool_module_reap(unsigned int slot, aotx_request *hold);

/* Host glue: name the directory of the module directories of the run. The loader opens a
 * module below it when the path the import carried does not open. */
void aotx_tool_module_root(const char *dir);

/* Host glue: read the plan, open each module file, check its digest and load it. The call
 * comes before the capture starts, because a capture takes no launch of its own. The
 * return is the modules the driver holds. */
unsigned int aotx_tool_module_open(void);

/* Host glue: add one node for each module the open loaded to the capture of the tick
 * graph. The return is the nodes added. */
unsigned int aotx_tool_module_capture(void *stream);

/* Host glue: give back every module the driver holds. */
void aotx_tool_module_close(void);

/* Host glue: the place of one entry in the plan, which is the node of that module. The
 * return is a negative number when the driver holds no module for that entry. */
int aotx_tool_module_place(unsigned int entry);

/* Host glue: the resource figures of the kernel of one module. They are the registers,
 * the local bytes, the threads of a block at the most, the version of the module text and
 * the architecture. The return is 0 when the figures are in hand. */
int aotx_tool_module_figures(unsigned int entry, int *regs, int *local, int *threads,
                             int *ptx, int *arch);

#endif
