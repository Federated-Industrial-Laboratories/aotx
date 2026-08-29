/* Purpose: Load the module file of every device tool and add its node to the tick graph.
 * Owns: The module handles the driver holds and the plan the device kernel wrote.
 * Launch shape: Host glue only; the module supplies the kernel of each node.
 * Lifetime: From the first capture to the close at the end of the run. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "tool/module.cuh"
#include "tool/module_host.h"

/* The disk-side library is C, and its header carries no C linkage guard of its own. The
 * system headers it needs therefore come first, and the header comes in with the guard. */
#include <signal.h>
#include <stdint.h>
extern "C" {
#include "disk/wire/diskwire.h"
}

/* One module the driver holds. The digest names the file it came from, so a capture that
 * finds the same digest keeps the module and pays no load. */
typedef struct aotx_tool_module_hold {
    CUmodule      module;
    CUfunction    function;
    unsigned int  entry;
    unsigned char digest[AOTX_SHA256_DIGEST];
    char          kernel[AOTX_CATALOG_NAME_BYTES];
    int           target;   /* the architecture of the target line of the module text */
    int           used;
} aotx_tool_module_hold;

static aotx_tool_module_hold aotx_tool_module_held[AOTX_TOOL_MODULES];
static aotx_tool_module_plan aotx_tool_module_read;
static unsigned int aotx_tool_module_count;

/* Give the entry the state REFUSED with the reason. The kernel writes the console line and
 * the bus note, so one path names every refusal of the catalog. */
static void aotx_tool_module_no(unsigned int entry, unsigned int why, const char *path)
{
    printf("module: %s is refused\n", path);
    aotx_tool_module_refuse<<<1, 1, 0, aotx_tool_module_line_of()>>>(entry, why);
    aotx_tool_module_wait();
}

/* Find a module the driver already holds for a digest and a kernel name. */
static int aotx_tool_module_same(const aotx_tool_module_row *row)
{
    for (unsigned int i = 0u; i < AOTX_TOOL_MODULES; ++i) {
        aotx_tool_module_hold *hold = &aotx_tool_module_held[i];
        if (hold->module == 0 || hold->used != 0) {
            continue;
        }
        if (memcmp(hold->digest, row->digest, AOTX_SHA256_DIGEST) == 0
            && strncmp(hold->kernel, row->kernel, AOTX_CATALOG_NAME_BYTES) == 0) {
            return (int)i;
        }
    }
    return -1;
}

/* Give back every module no row of this plan holds. */
static void aotx_tool_module_sweep(void)
{
    for (unsigned int i = 0u; i < AOTX_TOOL_MODULES; ++i) {
        aotx_tool_module_hold *hold = &aotx_tool_module_held[i];
        if (hold->module != 0 && hold->used == 0) {
            cuModuleUnload(hold->module);
            memset(hold, 0, sizeof *hold);
        }
    }
}

/* Read the module file of one row, check its digest and load it. The return is the place
 * of the module in the table, or -1 when the row is refused. */
static int aotx_tool_module_take(const aotx_tool_module_row *row, unsigned int at)
{
    char path[AOTX_MODULE_ROOT_BYTES + AOTX_CATALOG_NAME_BYTES + AOTX_TOOL_MODULE_FILE + 4u];
    unsigned char digest[AOTX_SHA256_DIGEST];
    aotx_sha256 state;
    size_t bytes = 0u;

    char *text = aotx_tool_module_text(row, &bytes, path, sizeof path);
    if (text == NULL) {
        aotx_tool_module_no(row->entry, AOTX_CATALOG_WHY_FILE, path);
        return -1;
    }
    /* The digest of the file is the digest the import carried. A file that changed on disk
     * after the import is refused, as a model file that changed is refused. */
    aotx_sha256_init(&state);
    aotx_sha256_update(&state, text, bytes);
    aotx_sha256_final(&state, digest);
    if (memcmp(digest, row->digest, AOTX_SHA256_DIGEST) != 0) {
        free(text);
        aotx_tool_module_no(row->entry, AOTX_CATALOG_WHY_DIGEST, path);
        return -1;
    }
    aotx_tool_module_hold *hold = &aotx_tool_module_held[at];
    if (cuModuleLoadData(&hold->module, text) != CUDA_SUCCESS) {
        free(text);
        hold->module = 0;
        aotx_tool_module_no(row->entry, AOTX_CATALOG_WHY_LOAD, path);
        return -1;
    }
    if (cuModuleGetFunction(&hold->function, hold->module, row->kernel) != CUDA_SUCCESS) {
        free(text);
        cuModuleUnload(hold->module);
        memset(hold, 0, sizeof *hold);
        aotx_tool_module_no(row->entry, AOTX_CATALOG_WHY_KERNEL, path);
        return -1;
    }
    hold->target = aotx_tool_module_target_of(text);
    free(text);
    memcpy(hold->digest, row->digest, AOTX_SHA256_DIGEST);
    snprintf(hold->kernel, sizeof hold->kernel, "%s", row->kernel);
    hold->entry = row->entry;
    return (int)at;
}

unsigned int aotx_tool_module_open(void)
{
    aotx_tool_module_wait();
    aotx_tool_module_scan<<<1, 1, 0, aotx_tool_module_line_of()>>>();
    aotx_check_runtime(cudaMemcpyFromSymbolAsync(&aotx_tool_module_read,
                                                 aotx_tool_module_list,
                                                 sizeof aotx_tool_module_read, 0,
                                                 cudaMemcpyDeviceToHost,
                                                 aotx_tool_module_line_of()),
                       "cudaMemcpyFromSymbolAsync");
    aotx_tool_module_wait();
    for (unsigned int i = 0u; i < AOTX_TOOL_MODULES; ++i) {
        aotx_tool_module_held[i].used = 0;
    }

    /* A row whose module the driver holds keeps it. Every other row reads its file, checks
     * the digest and loads it. A refused row leaves the plan, so the capture that follows
     * holds no node for it. */
    aotx_tool_module_plan *plan = &aotx_tool_module_read;
    unsigned int made = 0u;
    for (unsigned int r = 0u; r < plan->rows; ++r) {
        int at = aotx_tool_module_same(&plan->row[r]);
        if (at < 0) {
            unsigned int free_at = 0u;
            while (free_at < AOTX_TOOL_MODULES
                   && aotx_tool_module_held[free_at].module != 0) {
                free_at += 1u;
            }
            at = (free_at < AOTX_TOOL_MODULES)
               ? aotx_tool_module_take(&plan->row[r], free_at) : -1;
        }
        if (at < 0) {
            continue;
        }
        aotx_tool_module_held[at].used = 1;
        aotx_tool_module_held[at].entry = plan->row[r].entry;
        plan->row[made] = plan->row[r];
        made += 1u;
    }
    plan->rows = made;
    aotx_tool_module_sweep();
    aotx_tool_module_count = made;

    /* The device takes the plan that stands after the loads, so the entry of each node is
     * the entry the driver holds a module for. */
    aotx_check_runtime(cudaMemcpyToSymbolAsync(aotx_tool_module_list, plan, sizeof *plan,
                                               0, cudaMemcpyHostToDevice,
                                               aotx_tool_module_line_of()),
                       "cudaMemcpyToSymbolAsync");
    aotx_tool_module_bind<<<1, 1, 0, aotx_tool_module_line_of()>>>(made);
    aotx_tool_module_wait();
    return made;
}

/* The kernel of the module the loader holds for one entry, or zero. */
CUfunction aotx_tool_module_function(unsigned int entry)
{
    for (unsigned int i = 0u; i < AOTX_TOOL_MODULES; ++i) {
        aotx_tool_module_hold *hold = &aotx_tool_module_held[i];
        if (hold->module != 0 && hold->used != 0 && hold->entry == entry) {
            return hold->function;
        }
    }
    return 0;
}

int aotx_tool_module_target(unsigned int entry)
{
    for (unsigned int i = 0u; i < AOTX_TOOL_MODULES; ++i) {
        aotx_tool_module_hold *hold = &aotx_tool_module_held[i];
        if (hold->module != 0 && hold->used != 0 && hold->entry == entry) {
            return hold->target;
        }
    }
    return 0;
}

/* The modules the loader holds, and the entry of one of them in plan order. */
unsigned int aotx_tool_module_held_count(void)
{
    return aotx_tool_module_count;
}

unsigned int aotx_tool_module_entry_of(unsigned int at)
{
    return (at < aotx_tool_module_read.rows) ? aotx_tool_module_read.row[at].entry
                                             : (unsigned int)AOTX_CATALOG_NO_ENTRY;
}

/* The digest of one file, through the library of the disk side. The check program reads
 * the module file this way, as the feeder does before it publishes the head. */
/* Build the plan from the catalog and give one row of it back. The check program reads the
 * module file name and the kernel name this way, so no host code parses a manifest. */
int aotx_tool_module_plan_row(unsigned int at, aotx_tool_module_row *out)
{
    aotx_tool_module_wait();
    aotx_tool_module_scan<<<1, 1, 0, aotx_tool_module_line_of()>>>();
    aotx_check_runtime(cudaMemcpyFromSymbolAsync(&aotx_tool_module_read,
                                                 aotx_tool_module_list,
                                                 sizeof aotx_tool_module_read, 0,
                                                 cudaMemcpyDeviceToHost,
                                                 aotx_tool_module_line_of()),
                       "cudaMemcpyFromSymbolAsync");
    aotx_tool_module_wait();
    if (at >= aotx_tool_module_read.rows) {
        return 1;
    }
    *out = aotx_tool_module_read.row[at];
    return 0;
}

void aotx_tool_module_close(void)
{
    for (unsigned int i = 0u; i < AOTX_TOOL_MODULES; ++i) {
        if (aotx_tool_module_held[i].module != 0) {
            cuModuleUnload(aotx_tool_module_held[i].module);
        }
        memset(&aotx_tool_module_held[i], 0, sizeof aotx_tool_module_held[i]);
    }
    aotx_tool_module_count = 0u;
}
