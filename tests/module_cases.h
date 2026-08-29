/* Purpose: Give the module check its kernels, its call texts and its waits.
 * Owns: The call of each slot and the buffers the cases read back.
 * Threading: One host thread drives the cases; the kernels take one thread for each slot.
 * Lifetime: One run of the test program. */
#ifndef AOTX_TESTS_MODULE_CASES_H
#define AOTX_TESTS_MODULE_CASES_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "agent/agent_state.cuh"
#include "boot/check.h"
#include "catalog/catalog.cuh"
#include "sched/sched.cuh"
#include "tool/module.cuh"
#include "tool/module_host.h"
#include "tool/tool_state.cuh"

#include "catalog_feed.h"

/* Bytes of one call text of the check. A call names the tool and one argument. */
#define AOTX_MODULE_TEST_TEXT  512u

/* Ticks a case waits for the ring to empty. */
#define AOTX_MODULE_TEST_WAIT  600u

/* The call of each slot. The call stands in device state, as the gear of an agent holds
 * it, so no kernel keeps one in its frame. */
__device__ aotx_tool_call aotx_module_test_call[AOTX_SLOTS];

/* Parse one call text for each slot and open the request it names, as the agent step
 * does. The parse runs on the device, so the check exercises the parser of the run. */
__global__ void aotx_module_test_open(unsigned int count, const unsigned char *texts,
                                      const unsigned int *lengths, unsigned int *made,
                                      unsigned long long tick)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count || slot >= AOTX_SLOTS) {
        return;
    }
    aotx_tool_call *call = &aotx_module_test_call[slot];
    const unsigned char *text = texts + (unsigned long long)slot * AOTX_MODULE_TEST_TEXT;
    aotx_requests.slot[slot].request = 0u;
    aotx_tool_done[slot] = 0u;
    if (aotx_tool_parse(text, lengths[slot], call) == 0) {
        made[slot] = 0u;
        return;
    }
    made[slot] = aotx_tool_request(slot, call, 0u, tick);
}

/* Read the answer of a run of requests: the mark, the status, the length and the text. */
__global__ void aotx_module_test_read(unsigned int count, unsigned int *done,
                                      unsigned int *status, unsigned int *length,
                                      char *bytes)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= count || slot >= AOTX_SLOTS) {
        return;
    }
    done[slot] = aotx_tool_done[slot];
    status[slot] = aotx_requests.slot[slot].status;
    length[slot] = aotx_requests.slot[slot].result_len;
    for (unsigned int i = 0u; i < 64u; ++i) {
        bytes[(size_t)slot * 64u + i] = aotx_requests.slot[slot].result[i];
    }
}

/* Read the argument line one request holds. The check thus sees the key and the value the
 * device wrote for the ring and for the batch of the module. */
__global__ void aotx_module_test_line(unsigned int slot, unsigned int *length, char *bytes)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || slot >= AOTX_SLOTS) {
        return;
    }
    *length = aotx_requests.slot[slot].arg_len;
    for (unsigned int i = 0u; i < AOTX_TOOL_ARG_BYTES; ++i) {
        bytes[i] = aotx_requests.slot[slot].arg[i];
    }
}

/* Read one row of the batch of a module node, as the module reads it. */
__global__ void aotx_module_test_row(unsigned int node, unsigned int slot,
                                     unsigned int *take, unsigned int *length, char *value)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u || node >= (unsigned int)AOTX_TOOL_MODULES
        || slot >= AOTX_SLOTS) {
        return;
    }
    const aotx_tool_row *row =
        &aotx_tool_modules.row[(unsigned long long)node * AOTX_SLOTS + slot];
    *take = row->take;
    *length = row->argument[0].length;
    for (unsigned int i = 0u; i < 64u; ++i) {
        value[i] = row->argument[0].value[i];
    }
}

/* Give the request table and the agent table back empty. */
__global__ void aotx_module_test_free(void)
{
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < AOTX_SLOTS) {
        aotx_requests.slot[slot].request = 0u;
        aotx_requests.slot[slot].auth = AOTX_AUTH_NONE;
        aotx_requests.slot[slot].status = AOTX_TOOL_OK;
        aotx_requests.slot[slot].result_len = 0u;
        aotx_tool_done[slot] = 0u;
        aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_NONE;
        aotx_tool_embed.made[slot] = 0u;
        aotx_agents.agent[slot].state = AOTX_AGENT_STATE_FREE;
    }
}

/* Give the catalog back to the built-in tools, as a cold start holds it. */
__global__ void aotx_module_test_clear(void)
{
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    for (unsigned int i = 0u; i < AOTX_CATALOG_ARRIVING_MAX; ++i) {
        if (aotx_catalog.arriving[i].import != 0u) {
            aotx_catalog_free_run(aotx_catalog.arriving[i].run[0]);
            aotx_catalog_free_run(aotx_catalog.arriving[i].run[1]);
            aotx_catalog.arriving[i].import = 0u;
        }
    }
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        aotx_catalog_entry *row = &aotx_catalog.entry[i];
        if (row->state == AOTX_CATALOG_FREE
            || (row->kind == AOTX_MODULE_TOOL
                && row->tool.side == AOTX_CATALOG_SIDE_BUILT)) {
            continue;
        }
        aotx_catalog_release(row);
        row->state = AOTX_CATALOG_FREE;
        row->name_len = 0u;
        row->kind = 0u;
        row->why = AOTX_CATALOG_WHY_NONE;
    }
    aotx_catalog_anchor();
}

/* Device memory of a size, for a case that reads a table back. */
static void *aotx_module_test_take(size_t bytes)
{
    void *on = NULL;
    aotx_check_runtime(cudaMalloc(&on, bytes), "cudaMalloc");
    aotx_check_runtime(cudaMemset(on, 0, bytes), "cudaMemset");
    return on;
}

static void aotx_module_test_check(int good, const char *what, unsigned int *applied,
                                   unsigned int *failed)
{
    *applied += 1u;
    if (!good) {
        *failed += 1u;
        printf("module: FAILED %s\n", what);
    }
}

/* Run ticks until the apply took every record of the ring. */
static void aotx_module_test_settle(aotx_pump *pump, aotx_seam_rings *rings)
{
    aotx_inbound_preamble *preamble = (aotx_inbound_preamble *)rings->inbound_map;
    for (unsigned int i = 0u; i < AOTX_MODULE_TEST_WAIT; ++i) {
        aotx_pump_tick(pump);
        unsigned long long head = __atomic_load_n(&preamble->head, __ATOMIC_ACQUIRE);
        unsigned long long took = __atomic_load_n(&preamble->consumed, __ATOMIC_ACQUIRE);
        if (took >= head) {
            aotx_pump_tick(pump);
            return;
        }
    }
}

/* Read one module directory from disk and put the digest of its module file in the head.
 * The records then go to the inbound ring, as the feeder puts them there. */
static int aotx_module_test_import(aotx_seam_rings *rings, const char *dir,
                                   unsigned int import, unsigned long long boot_id,
                                   unsigned int replayed)
{
    aotx_test_module module;
    char path[1024];
    char file[256];
    if (aotx_test_module_dir(&module, dir) != 0) {
        return 1;
    }
    if (aotx_test_manifest_value(module.manifest, module.manifest_len, "module", file,
                                 sizeof file) != 0 && file[0] != '\0') {
        snprintf(path, sizeof path, "%s/%s", dir, file);
        if (aotx_tool_module_digest(path, module.digest) != 0) {
            aotx_test_module_free(&module);
            return 1;
        }
    }
    unsigned char *bodies = (unsigned char *)calloc(AOTX_TEST_IMPORT_MAX, AOTX_BODY_BYTES);
    unsigned int *sizes = (unsigned int *)calloc(AOTX_TEST_IMPORT_MAX,
                                                 sizeof(unsigned int));
    unsigned int made = aotx_test_import_build(&module, import, bodies, sizes);
    unsigned int flags = (replayed != 0u) ? (unsigned int)AOTX_FLAG_REPLAYED : 0u;
    for (unsigned int i = 0u; i < made; ++i) {
        aotx_test_feed_records(rings, AOTX_REC_IMPORT, AOTX_CLASS_A, AOTX_WRITER_FEEDER,
                               flags, bodies + (size_t)i * AOTX_BODY_BYTES, sizes[i], 1u,
                               boot_id);
    }
    free(bodies);
    free(sizes);
    aotx_test_module_free(&module);
    return 0;
}

/* Build one call text for a slot. Every slot gets a text of its own count of words, so
 * the answer of every row differs from the answer of every other row. */
static unsigned int aotx_module_test_text(char *out, unsigned int bytes,
                                          const char *tool, unsigned int slot)
{
    char words[AOTX_MODULE_TEST_TEXT];
    unsigned int at = 0u;
    for (unsigned int w = 0u; w <= slot && at + 2u < sizeof words; ++w) {
        if (at != 0u) {
            words[at] = ' ';
            at += 1u;
        }
        words[at] = (char)('a' + (char)(w % 26u));
        at += 1u;
    }
    words[at] = '\0';
    return (unsigned int)snprintf(out, bytes,
                                  "<tool_call>\n{\"name\": \"%s\", \"arguments\": "
                                  "{\"text\": \"%s\"}}\n</tool_call>", tool, words);
}

/* The entry of an installed tool of a name, and the state of any entry of that name. */
static unsigned int aotx_module_test_entry(const char *name, unsigned int *state,
                                           unsigned int *why, unsigned int *import)
{
    aotx_catalog_state *held = aotx_test_catalog_read();
    unsigned int found = AOTX_MODULE_SLOTS;
    unsigned int length = (unsigned int)strlen(name);
    *state = AOTX_CATALOG_FREE;
    *why = AOTX_CATALOG_WHY_NONE;
    *import = 0u;
    for (unsigned int i = 0u; i < AOTX_MODULE_SLOTS; ++i) {
        if (held->entry[i].state != AOTX_CATALOG_FREE
            && held->entry[i].name_len == length
            && strncmp(held->entry[i].name, name, length) == 0) {
            found = i;
            *state = held->entry[i].state;
            *why = held->entry[i].why;
            *import = held->entry[i].import;
            break;
        }
    }
    free(held);
    return found;
}

/* The tail of the module state: the entry of each node and the counters. The state holds
 * megabytes of scratch in front of it, so the check reads the tail alone. The size check
 * below fails when a field joins the state and this copy does not follow it. */
typedef struct aotx_module_test_tail {
    unsigned int entry[AOTX_TOOL_MODULES];
    unsigned int nodes;
    unsigned int gen;
    unsigned int took;
    unsigned int gave;
    unsigned int over;
    unsigned int untaken;
    unsigned int bare;
    unsigned int captures;
    unsigned int before;
    unsigned int after;
    unsigned int took_us;
} aotx_module_test_tail;

typedef char aotx_module_test_tail_check[
    (offsetof(aotx_module_test_tail, took_us)
     == offsetof(aotx_tool_module_state, took_us)
        - offsetof(aotx_tool_module_state, entry))
    ? 1 : -1];

static void aotx_module_test_tail_read(aotx_module_test_tail *out)
{
    aotx_check_runtime(cudaMemcpyFromSymbol(out, aotx_tool_modules, sizeof *out,
                                            offsetof(aotx_tool_module_state, entry)),
                       "cudaMemcpyFromSymbol");
}

#endif
