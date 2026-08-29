#define _GNU_SOURCE
/* Purpose: Check the device tool modules: the node, the batch, the capture and the digest.
 * Owns: The counts of the cases and the buffers each case reads back.
 * Launch shape: The cases drive the tick graph; the kernels take one thread for each slot.
 * Lifetime: One run of the test program.
 *
 * The check takes one module directory that holds a built module file. The build makes
 * that directory from sdk/examples/word_count with tests/module_setup.sh. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda.h>

#include "mem/mem.cuh"
#include "seam/seam.cuh"

#include "module_cases.h"

static unsigned int aotx_module_applied;
static unsigned int aotx_module_failed;

static void aotx_module_case(int good, const char *what)
{
    aotx_module_test_check(good, what, &aotx_module_applied, &aotx_module_failed);
}

/* Open one request for each of a run of slots, wait two ticks, and read the answers. The
 * module node stands between the fill and the tool step. A request that opens after the
 * step of one tick is therefore answered in the tick after it. */
static void aotx_module_test_batch(unsigned int node);

static void aotx_module_test_run(aotx_pump *pump, const char *tool, unsigned int count,
                                 unsigned int look)
{
    char *texts = (char *)calloc(AOTX_SLOTS, AOTX_MODULE_TEST_TEXT);
    unsigned int *lengths = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    unsigned char *on_texts =
        (unsigned char *)aotx_module_test_take((size_t)AOTX_SLOTS * AOTX_MODULE_TEST_TEXT);
    unsigned int *on_lengths =
        (unsigned int *)aotx_module_test_take(AOTX_SLOTS * sizeof(unsigned int));
    unsigned int *on_made =
        (unsigned int *)aotx_module_test_take(AOTX_SLOTS * sizeof(unsigned int));

    for (unsigned int i = 0u; i < count; ++i) {
        lengths[i] = aotx_module_test_text(texts + (size_t)i * AOTX_MODULE_TEST_TEXT,
                                           AOTX_MODULE_TEST_TEXT, tool, i);
    }
    aotx_check_runtime(cudaMemcpy(on_texts, texts,
                                  (size_t)AOTX_SLOTS * AOTX_MODULE_TEST_TEXT,
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(on_lengths, lengths, AOTX_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_module_test_free<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_module_test_open<<<1, AOTX_SLOTS>>>(count, on_texts, on_lengths, on_made, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    unsigned int *made = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(made, on_made, AOTX_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    unsigned int opened = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        opened += (made[i] != 0u) ? 1u : 0u;
    }
    aotx_module_case(opened == count, "every call of the run opened a request");

    /* The fill of the first tick gives the module its rows and the module answers in that
     * tick. The fill of the tick after it takes the rows back, because the answer is in
     * hand. A case that reads a row therefore reads it between the two ticks. */
    aotx_pump_tick(pump);
    if (look != 0u) {
        aotx_module_test_batch(0u);
    }
    aotx_pump_tick(pump);

    unsigned int *on_done = (unsigned int *)aotx_module_test_take(AOTX_SLOTS * 4u);
    unsigned int *on_status = (unsigned int *)aotx_module_test_take(AOTX_SLOTS * 4u);
    unsigned int *on_length = (unsigned int *)aotx_module_test_take(AOTX_SLOTS * 4u);
    char *on_bytes = (char *)aotx_module_test_take((size_t)AOTX_SLOTS * 64u);
    aotx_module_test_read<<<1, AOTX_SLOTS>>>(count, on_done, on_status, on_length,
                                             on_bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *done = (unsigned int *)calloc(AOTX_SLOTS, 4u);
    unsigned int *status = (unsigned int *)calloc(AOTX_SLOTS, 4u);
    unsigned int *length = (unsigned int *)calloc(AOTX_SLOTS, 4u);
    char *bytes = (char *)calloc(AOTX_SLOTS, 64u);
    aotx_check_runtime(cudaMemcpy(done, on_done, AOTX_SLOTS * 4u, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(status, on_status, AOTX_SLOTS * 4u,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(length, on_length, AOTX_SLOTS * 4u,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(bytes, on_bytes, (size_t)AOTX_SLOTS * 64u,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");

    unsigned int answered = 0u;
    unsigned int right = 0u;
    for (unsigned int i = 0u; i < count; ++i) {
        char want[64];
        snprintf(want, sizeof want, "%u %s", i + 1u, (i == 0u) ? "word" : "words");
        answered += (done[i] != 0u && status[i] == AOTX_TOOL_OK) ? 1u : 0u;
        if (length[i] == (unsigned int)strlen(want)
            && strncmp(bytes + (size_t)i * 64u, want, strlen(want)) == 0) {
            right += 1u;
        }
    }
    printf("module: at %u rows %u answered and %u gave the count of their own text\n",
           count, answered, right);
    aotx_module_case(answered == count, "every request of the run was answered");
    aotx_module_case(right == count, "every row gave the count of its own text");

    free(texts);
    free(lengths);
    free(made);
    free(done);
    free(status);
    free(length);
    free(bytes);
    cudaFree(on_texts);
    cudaFree(on_lengths);
    cudaFree(on_made);
    cudaFree(on_done);
    cudaFree(on_status);
    cudaFree(on_length);
    cudaFree(on_bytes);
}

/* The argument line of a request: the keys of the manifest and their values, parted by the
 * unit separator byte. The record of a host tool and the batch of a module read that line. */
static void aotx_module_test_arguments(void)
{
    unsigned int *on_length = (unsigned int *)aotx_module_test_take(4u);
    char *on_bytes = (char *)aotx_module_test_take(AOTX_TOOL_ARG_BYTES);
    unsigned int length = 0u;
    char bytes[AOTX_TOOL_ARG_BYTES + 1];
    aotx_module_test_line<<<1, 1>>>(0u, on_length, on_bytes);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&length, on_length, 4u, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(bytes, on_bytes, AOTX_TOOL_ARG_BYTES,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    bytes[AOTX_TOOL_ARG_BYTES] = '\0';
    printf("module: the argument line of slot 0 is \"%.*s\" of %u bytes\n", (int)length,
           bytes, length);
    aotx_module_case(length == 6u && strncmp(bytes, "text=a", 6u) == 0,
                     "the argument line holds the key, the sign and the value");
    cudaFree(on_length);
    cudaFree(on_bytes);
}

/* The row of the batch the fill wrote for the module. */
static void aotx_module_test_batch(unsigned int node)
{
    unsigned int *on_take = (unsigned int *)aotx_module_test_take(4u);
    unsigned int *on_length = (unsigned int *)aotx_module_test_take(4u);
    char *on_value = (char *)aotx_module_test_take(64u);
    unsigned int take = 0u;
    unsigned int length = 0u;
    char value[65];
    aotx_module_test_row<<<1, 1>>>(node, 0u, on_take, on_length, on_value);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(&take, on_take, 4u, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(&length, on_length, 4u, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(value, on_value, 64u, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    value[64] = '\0';
    aotx_module_case(take == 1u && length == 1u && value[0] == 'a',
                     "the fill gave the module the value of the key of its row");
    cudaFree(on_take);
    cudaFree(on_length);
    cudaFree(on_value);
}

/* Append one byte to the module file, so its digest is not the digest of the import. */
static int aotx_module_test_touch(const char *dir, const char *file, int put)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, file);
    if (put != 0) {
        FILE *at = fopen(path, "ab");
        if (at == NULL) {
            return 1;
        }
        fputc('\n', at);
        fclose(at);
        return 0;
    }
    /* Take the byte off again, so the case that follows reads the file of the import. */
    FILE *at = fopen(path, "rb");
    if (at == NULL) {
        return 1;
    }
    fseek(at, 0, SEEK_END);
    long size = ftell(at);
    fclose(at);
    return (size > 0 && truncate(path, size - 1) == 0) ? 0 : 1;
}

int main(int argc, char **argv)
{
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    CUdevice device;
    CUcontext context;
    aotx_module_test_tail tail;

    if (argc < 2) {
        printf("usage: aotx_module_device_test <module-directory>\n");
        return 2;
    }
    const char *dir = argv[1];
    unsigned long long boot_id = 0x0D0D01Eull;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device),
                      "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("module: the map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    aotx_seam_note_boot<<<1, 1>>>(0ull, 0ull);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    if (aotx_pump_build(&pump, 0ull, 1u) != 0) {
        printf("module: the tick graph did not build\n");
        return 1;
    }
    unsigned int base = pump.nodes;
    printf("module: the tick graph holds %u nodes with no device tool\n", base);
    aotx_module_case(pump.modules == 0u, "a run with no module holds no module node");

    /* The head of an import carries the tail of the path of its directory, and that field
     * holds 63 bytes. The check names the directory it was given as the root, so the
     * loader opens the module file whatever the length of the path. */
    aotx_tool_module_root(dir);

    /* The import of a device tool through the inbound ring and the apply node. */
    if (aotx_module_test_import(&rings, dir, 1u, boot_id, 0u) != 0) {
        printf("module: the module directory %s did not read\n", dir);
        return 1;
    }
    aotx_module_test_settle(&pump, &rings);
    unsigned int state = 0u;
    unsigned int why = 0u;
    unsigned int import = 0u;
    unsigned int entry = aotx_module_test_entry("word_count", &state, &why, &import);
    aotx_module_case(entry < AOTX_MODULE_SLOTS && state == AOTX_CATALOG_INSTALLED,
                     "the import of the device tool installed an entry");
    aotx_module_case(import == 1u, "the entry holds the number of the import");
    if (state != AOTX_CATALOG_INSTALLED) {
        printf("module: the import was refused with the reason %u\n", why);
        return 1;
    }

    /* The capture that follows the import holds one node more. */
    aotx_module_test_tail_read(&tail);
    printf("module: the tick graph holds %u nodes with one device tool, and the capture "
           "took %u microseconds\n", pump.nodes, tail.took_us);
    aotx_module_case(pump.modules == 1u, "the graph holds one node for the module");
    aotx_module_case(pump.nodes == base + 1u, "the node count grew by one");
    aotx_module_case(pump.recaptures >= 1u, "the pump captured the graph again");
    aotx_module_case(tail.captures >= 1u, "the record of the capture was written");
    aotx_module_case(tail.before == base && tail.after == base + 1u,
                     "the record names the node count before and after");
    aotx_module_case(tail.nodes == 1u, "the module state holds one node");
    aotx_module_case(tail.entry[0] == entry, "the node names the entry of the module");

    /* A request at one row and at the row count of the profile. */
    aotx_module_test_run(&pump, "word_count", 1u, 1u);
    aotx_module_test_arguments();
    aotx_module_test_run(&pump, "word_count", AOTX_SLOTS, 0u);
    aotx_module_test_tail_read(&tail);
    printf("module: the modules took %u rows and gave %u answers, %u over the bound and "
           "%u untaken\n", tail.took, tail.gave, tail.over, tail.untaken);
    aotx_module_case(tail.over == 0u, "no answer went past the bound of a result");
    aotx_module_case(tail.untaken == 0u, "no untaken row was written");

    /* The restore. The catalog goes back to the built-in tools and the records of the
     * import come again with the replayed flag. The module then loads by its digest. */
    aotx_module_test_clear<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_module_test_settle(&pump, &rings);
    aotx_module_case(pump.modules == 0u, "the graph holds no module node after the clear");
    if (aotx_module_test_import(&rings, dir, 2u, boot_id, 1u) != 0) {
        printf("module: the replay of the import did not read the directory\n");
        return 1;
    }
    aotx_module_test_settle(&pump, &rings);
    entry = aotx_module_test_entry("word_count", &state, &why, &import);
    aotx_module_case(state == AOTX_CATALOG_INSTALLED,
                     "the replay of the import installed the entry again");
    aotx_module_case(import == 2u, "the entry holds the number of the import that replayed");
    aotx_module_case(pump.modules == 1u, "the module loaded again by its digest");
    aotx_module_test_run(&pump, "word_count", 1u, 0u);

    /* The digest refusal: the module file changes on disk and the capture refuses it. */
    aotx_test_module module;
    char file[256];
    file[0] = '\0';
    if (aotx_test_module_dir(&module, dir) == 0) {
        aotx_test_manifest_value(module.manifest, module.manifest_len, "module", file,
                                 sizeof file);
        aotx_test_module_free(&module);
    }
    aotx_module_case(file[0] != '\0', "the manifest names the module file");
    if (file[0] != '\0' && aotx_module_test_touch(dir, file, 1) == 0) {
        /* The loader keeps a module whose digest and kernel name it already holds, so a
         * capture inside a run pays no load. A run that starts again holds no module, and
         * it reads every module file. The close puts the loader in that state. */
        aotx_tool_module_close();
        aotx_pump_recapture(&pump);
        entry = aotx_module_test_entry("word_count", &state, &why, &import);
        aotx_module_case(state == AOTX_CATALOG_REFUSED,
                         "a module file that changed on disk is refused");
        aotx_module_case(why == AOTX_CATALOG_WHY_DIGEST,
                         "the reason of the refusal names the digest");
        aotx_module_case(pump.modules == 0u, "the graph holds no node for a refused module");
        aotx_module_case(pump.nodes == base, "the node count fell back to the list");
        aotx_module_test_touch(dir, file, 0);
    } else {
        aotx_module_case(0, "the module file took a byte for the digest case");
    }

    aotx_pump_close(&pump);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    printf("module: %u cases applied, %u failed\n", aotx_module_applied,
           aotx_module_failed);
    return (aotx_module_failed == 0u) ? 0 : 1;
}
