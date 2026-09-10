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

#include "boot/boot.cuh"
#include "mem/mem.cuh"
#include "seam/seam.cuh"

#include "module_cases.h"
#include "wrap_fixture.h"

/* Ticks the seam case waits for a disk-side program to answer. The feeder polls, so an
 * answer takes a run of ticks and not one. */
#define AOTX_MODULE_SEAM_TICKS 4000u

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
    /* The separator comes before every pair, the first one included, so the line the
     * feeder reads starts with that byte. A line that does not is one bare value. */
    aotx_module_case(length == 7u && bytes[0] == AOTX_TOOL_UNIT
                     && strncmp(bytes + 1, "text=a", 6u) == 0,
                     "the argument line starts with the separator and holds the pair");
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

/* The seam case. The request the device writes goes to the disk side through the real
 * drain, and the real feeder answers it. A built-in host tool gives the bytes of a file. A
 * host tool that came in as a module gives the output of its program. The case therefore
 * reads the argument line the way a run reads it, and a line the feeder cannot split
 * fails it.
 *
 * The pattern is the loop arm of the feed check. The check starts the two programs of the
 * disk side and drives the tick graph while they run. */
static void aotx_module_test_seam(aotx_pump *pump, aotx_seam_rings *rings,
                                  const char *build, const char *modules)
{
    aotx_boot_children children;
    char journal[1024];
    char root[1024];
    char tools[1024];
    char line[AOTX_MODULE_TEST_TEXT];
    char answer[AOTX_TOOL_RESULT_BYTES + 1];
    const char *content = "the seam carries these bytes";
    const char *file_reply =
        "sha256: ea506ad830605864c3607c200e4946852f8ff6ca0963981fffa2bc83018bfd23\n"
        "the seam carries these bytes";
    unsigned int status = 0u;
    unsigned int length = 0u;

    memset(&children, 0, sizeof children);
    snprintf(journal, sizeof journal, "%s/module-seam", build);
    snprintf(root, sizeof root, "%s/module-seam-root", build);
    snprintf(tools, sizeof tools, "%s/module-seam-tools", build);
    snprintf(line, sizeof line, "rm -rf %s %s %s", journal, root, tools);
    if (system(line) != 0) {
        aotx_module_case(0, "the directories of the seam case were made");
        return;
    }
    mkdir(journal, 0777);
    if (aotx_module_test_file(root, "hello.txt", content) != 0
        || aotx_module_test_copy(modules, tools, "echo_upper") != 0) {
        aotx_module_case(0, "the file and the module of the seam case were made");
        return;
    }
    if (aotx_boot_start_drain(&children, rings, journal, NULL, NULL) != 0
        || aotx_boot_start_feed(&children, rings, -1, root, journal, NULL, tools, 0) != 0) {
        aotx_module_case(0, "the drain and the feeder of the seam case started");
        aotx_boot_stop(&children);
        return;
    }

    /* The feeder publishes the import of the module, and the apply of a tick takes it. */
    unsigned int state = 0u;
    unsigned int why = 0u;
    unsigned int import = 0u;
    unsigned int entry = AOTX_MODULE_SLOTS;
    for (unsigned int i = 0u; i < AOTX_MODULE_SEAM_TICKS && entry >= AOTX_MODULE_SLOTS;
         ++i) {
        aotx_pump_tick(pump);
        entry = aotx_module_test_entry("echo_upper", &state, &why, &import);
    }
    aotx_module_case(entry < AOTX_MODULE_SLOTS && state == AOTX_CATALOG_INSTALLED,
                     "the feeder imported the host tool of the seam case");
    if (entry >= AOTX_MODULE_SLOTS) {
        aotx_seam_finish(rings);
        aotx_boot_stop(&children);
        return;
    }

    /* The built-in host tool: the reply carries the digest before the file bytes. */
    aotx_module_test_free<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    snprintf(line, sizeof line,
             "<tool_call>\n{\"name\": \"fs_read\", \"arguments\": "
             "{\"path\": \"hello.txt\"}}\n</tool_call>");
    aotx_module_case(aotx_module_test_call_one(0u, line) != 0u,
                     "the call of the built-in host tool opened a request");
    int came = aotx_module_test_answer(pump, 0u, AOTX_MODULE_SEAM_TICKS, &status, &length,
                                       answer, AOTX_TOOL_RESULT_BYTES);
    printf("module: the file tool gave status %u and %u bytes: %.60s\n", status, length,
           came ? answer : "");
    aotx_module_case(came != 0 && status == AOTX_TOOL_OK,
                     "the feeder answered the built-in host tool");
    aotx_module_case(came != 0 && length == (unsigned int)strlen(file_reply)
                     && memcmp(answer, file_reply, strlen(file_reply)) == 0,
                     "the reply carries the digest first and then the file");

    /* The host tool that came in as a module: the reply carries the output of the
     * program. The feeder finds that program under the number of the import. The call
     * takes the slot after the one before it. The number of a request comes from the slot
     * and the count of the requests that slot made. A slot given back would give the same
     * number again, and the feeder answers a number one time. */
    snprintf(line, sizeof line,
             "<tool_call>\n{\"name\": \"echo_upper\", \"arguments\": "
             "{\"text\": \"one two three\"}}\n</tool_call>");
    aotx_module_case(aotx_module_test_call_one(1u, line) != 0u,
                     "the call of the module host tool opened a request");
    came = aotx_module_test_answer(pump, 1u, AOTX_MODULE_SEAM_TICKS, &status, &length,
                                   answer, AOTX_TOOL_RESULT_BYTES);
    printf("module: the program gave status %u and %u bytes: %.60s\n", status, length,
           came ? answer : "");
    aotx_module_case(came != 0 && status == AOTX_TOOL_OK,
                     "the feeder ran the program of the module host tool");
    aotx_module_case(came != 0 && strncmp(answer, "ONE TWO THREE", 13u) == 0,
                     "the reply carries the output of the program");
    /* The two programs end when the rings close, as they end at the close of a run. */
    aotx_seam_finish(rings);
    aotx_boot_stop(&children);
}

int main(int argc, char **argv)
{
    aotx_mem_map map;
    aotx_seam_rings rings;
    aotx_pump pump;
    CUdevice device;
    CUcontext context;
    aotx_module_test_tail tail;

    if (argc < 3) {
        printf("usage: aotx_module_device_test <module-directory> <build-directory>\n");
        return 2;
    }
    const char *dir = argv[1];
    const char *build = argv[2];
    unsigned long long boot_id = 0x0D0D01Eull;
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device),
                      "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    aotx_test_call_upload(AOTX_CALL_HERMES);
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
     * holds 63 bytes. The root of the loader is the directory that holds the module
     * directories, which is the one above the directory given. */
    char root[1024];
    const char *cut = strrchr(dir, '/');
    snprintf(root, sizeof root, "%.*s", (int)((cut != NULL) ? (cut - dir) : 1),
             (cut != NULL) ? dir : ".");
    aotx_tool_module_root(root);

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

    /* The seam case runs last, because it starts the two programs of the disk side and
     * gives them the rings of this run. */
    char examples[1024];
    snprintf(examples, sizeof examples, "%.*s/echo_upper",
             (int)(strrchr(dir, '/') != NULL ? strrchr(dir, '/') - dir : 0), dir);
    aotx_module_test_seam(&pump, &rings, build, examples);

    aotx_pump_close(&pump);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    printf("module: %u cases applied, %u failed\n", aotx_module_applied,
           aotx_module_failed);
    return (aotx_module_failed == 0u) ? 0 : 1;
}
