/* Purpose: Check that one module directory compiles, loads and answers a batch.
 * Owns: The context of the check, the device buffers and the counts of the checks.
 * Launch shape: Host glue only; the kernels of the check judge every figure.
 * Lifetime: One run of the check program. */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "boot/check.h"
#include "catalog/check.cuh"
#include "mem/mem.cuh"
#include "tool/module_host.h"
#include "seam/seam.cuh"

/* The tick budget of a run with no decode, in microseconds. A module node stands inside
 * one tick, so a launch that takes more than the budget takes the tick with it. */
#define AOTX_CHECK_BUDGET_US 10000u

static unsigned int aotx_check_applied;
static unsigned int aotx_check_failed;

int aotx_check_program(const char *dir, const aotx_check_entry *entry, const char *name,
                       unsigned int *applied, unsigned int *failed);

static void aotx_check_say(int ok, const char *what, unsigned long long figure)
{
    aotx_check_applied += 1u;
    aotx_check_failed += ok ? 0u : 1u;
    printf("check: %s %s %llu\n", ok ? "ok  " : "FAIL", what, figure);
}

/* Read a whole file. The caller frees the bytes. */
static unsigned char *aotx_check_read(const char *path, unsigned int *length)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char *bytes = (size >= 0) ? (unsigned char *)malloc((size_t)size + 1u) : NULL;
    if (bytes == NULL || fread(bytes, 1u, (size_t)size, file) != (size_t)size) {
        free(bytes);
        fclose(file);
        return NULL;
    }
    bytes[size] = '\0';
    *length = (unsigned int)size;
    fclose(file);
    return bytes;
}

/* Give a device text that ends with a zero byte, of a fixed width. */
static char *aotx_check_text(const char *from, unsigned int width)
{
    char *host = (char *)calloc(width, 1u);
    char *on = NULL;
    snprintf(host, width, "%s", from);
    cudaMalloc((void **)&on, width);
    cudaMemcpy(on, host, width, cudaMemcpyHostToDevice);
    free(host);
    return on;
}

/* The last component of a path, which is the name of the module. */
static const char *aotx_check_last(const char *path)
{
    const char *tail = strrchr(path, '/');
    return (tail != NULL && tail[1] != '\0') ? tail + 1 : path;
}

/* Import the manifest of a directory through the reader of the device and read the entry
 * back. The return is the entry, or AOTX_MODULE_SLOTS. */
static unsigned int aotx_check_import_dir(const char *dir, aotx_check_entry *out,
                                          aotx_check_entry *on)
{
    char path[1024];
    unsigned int length = 0u;
    snprintf(path, sizeof path, "%s/module.manifest", dir);
    unsigned char *bytes = aotx_check_read(path, &length);
    if (bytes == NULL) {
        printf("check: FAIL the directory holds no module.manifest: %s\n", dir);
        return AOTX_MODULE_SLOTS;
    }
    unsigned char *on_bytes = NULL;
    cudaMalloc((void **)&on_bytes, length + 1u);
    cudaMemcpy(on_bytes, bytes, length, cudaMemcpyHostToDevice);
    free(bytes);
    char *on_name = aotx_check_text(aotx_check_last(dir), AOTX_IMPORT_NAME_BYTES);
    char *on_path = aotx_check_text(dir, AOTX_IMPORT_PATH_BYTES);
    aotx_check_import<<<1, 1>>>(on_bytes, length, AOTX_MODULE_TOOL, on_name, on_path);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(on_bytes);
    cudaFree(on_name);
    cudaFree(on_path);

    /* The entry of the built-in tools stands before this one, so the import takes the
     * first free entry after them. */
    unsigned int at = AOTX_CATALOG_BUILT_IN;
    aotx_check_report<<<1, 1>>>(at, on);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(out, on, sizeof *out, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    return at;
}

/* Run the module over a batch of a count of rows and judge what it wrote. */
static void aotx_check_batch(unsigned int node, unsigned int entry, unsigned int rows)
{
    aotx_check_verdict verdict;
    cudaEvent_t start;
    cudaEvent_t stop;
    float took = 0.0f;
    char line[128];

    aotx_check_fill<<<AOTX_SLOTS, 1>>>(node, entry, rows);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaEventCreate(&start), "cudaEventCreate");
    aotx_check_runtime(cudaEventCreate(&stop), "cudaEventCreate");
    aotx_check_runtime(cudaEventRecord(start, 0), "cudaEventRecord");
    aotx_tool_module_launch(entry);
    aotx_check_runtime(cudaEventRecord(stop, 0), "cudaEventRecord");
    aotx_check_runtime(cudaEventSynchronize(stop), "cudaEventSynchronize");
    aotx_check_runtime(cudaEventElapsedTime(&took, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    aotx_check_judge<<<AOTX_SLOTS, 1>>>(rows);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpyFromSymbol(&verdict, aotx_check_out, sizeof verdict),
                       "cudaMemcpyFromSymbol");

    unsigned int micro = (unsigned int)(took * 1000.0f);
    snprintf(line, sizeof line, "at %u rows the module answered rows:", rows);
    aotx_check_say(verdict.done == rows, line, verdict.done);
    snprintf(line, sizeof line, "at %u rows a status the contract refuses:", rows);
    aotx_check_say(verdict.status_bad == 0u, line, verdict.status_bad);
    snprintf(line, sizeof line, "at %u rows a length over the bound:", rows);
    aotx_check_say(verdict.over == 0u, line, verdict.over);
    snprintf(line, sizeof line, "at %u rows an untaken row that was written:", rows);
    aotx_check_say(verdict.untaken == 0u, line, verdict.untaken);
    snprintf(line, sizeof line, "at %u rows the longest result of %u bytes:", rows,
             (unsigned int)AOTX_TOOL_RESULT_BYTES);
    aotx_check_say(verdict.longest <= (unsigned int)AOTX_TOOL_RESULT_BYTES, line,
                   verdict.longest);
    snprintf(line, sizeof line, "at %u rows the launch of %u microseconds took:", rows,
             AOTX_CHECK_BUDGET_US);
    aotx_check_say(micro <= AOTX_CHECK_BUDGET_US, line, micro);
}

/* The device arm: load the module by digest, read its figures and run the batches. */
static void aotx_check_device(const char *dir, unsigned int entry, unsigned int rows)
{
    unsigned char digest[32];
    unsigned char *on_digest = NULL;
    char path[1024];
    aotx_tool_module_row row;
    int regs = 0;
    int local = 0;
    int threads = 0;
    int ptx = 0;
    int arch = 0;

    if (aotx_tool_module_plan_row(0u, &row) != 0) {
        aotx_check_say(0, "the plan holds a row for the module:", 0ull);
        return;
    }
    snprintf(path, sizeof path, "%s/%s", dir, row.file);
    if (aotx_tool_module_digest(path, digest) != 0) {
        aotx_check_say(0, "the module file opens and hashes:", 0ull);
        return;
    }
    cudaMalloc((void **)&on_digest, sizeof digest);
    cudaMemcpy(on_digest, digest, sizeof digest, cudaMemcpyHostToDevice);
    aotx_check_digest<<<1, 1>>>(entry, on_digest);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(on_digest);

    unsigned int made = aotx_tool_module_open();
    aotx_check_say(made == 1u, "modules the driver holds:", made);
    if (made != 1u) {
        return;
    }
    int node = aotx_tool_module_place(entry);
    if (node < 0 || aotx_tool_module_figures(entry, &regs, &local, &threads, &ptx,
                                             &arch) != 0) {
        aotx_check_say(0, "the figures of the kernel:", 0ull);
        return;
    }
    aotx_check_say(local == 0, "bytes of local memory, which must be none:",
                   (unsigned long long)local);
    aotx_check_say(regs > 0, "registers the kernel keeps:", (unsigned long long)regs);
    aotx_check_say(threads >= (int)AOTX_TOOL_MODULE_THREADS,
                   "threads of a block the kernel takes:", (unsigned long long)threads);
    aotx_check_say(arch <= (int)AOTX_ARCH, "the architecture of the module:",
                   (unsigned long long)arch);
    aotx_check_say(ptx > 0, "the version of the module text, times ten:",
                   (unsigned long long)ptx);
    if (rows != 0u) {
        aotx_check_batch((unsigned int)node, entry, rows);
        return;
    }
    aotx_check_batch((unsigned int)node, entry, 1u);
    aotx_check_batch((unsigned int)node, entry, AOTX_SLOTS);
}

int main(int argc, char **argv)
{
    aotx_check_entry held;
    aotx_check_entry *on = NULL;
    aotx_mem_map map;
    aotx_seam_rings rings;
    CUdevice device;
    CUcontext context;

    if (argc < 2) {
        printf("usage: aotx_module_check <directory> [rows]\n");
        return 2;
    }
    const char *dir = argv[1];
    unsigned int rows = (argc > 2) ? (unsigned int)strtoul(argv[2], NULL, 10) : 0u;
    if (rows > AOTX_SLOTS) {
        printf("check: the row count is 1 to %u\n", (unsigned int)AOTX_SLOTS);
        return 2;
    }
    aotx_check_driver(cuInit(0), "cuInit");
    aotx_check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    aotx_check_driver(cuDevicePrimaryCtxRetain(&context, device),
                      "cuDevicePrimaryCtxRetain");
    aotx_check_driver(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    /* The import writes a console line and a bus note, so the rings of the run stand
     * before the catalog takes a record. The check reads no block of them. */
    unsigned long long boot_id = 0xC4EC4ull;
    if (aotx_mem_reserve(&map) != 0 || aotx_seam_open(&rings, boot_id) != 0) {
        printf("check: the memory map or the rings did not open\n");
        return 1;
    }
    aotx_seam_bind(&rings, map.ring, map.ring_bytes, boot_id);
    aotx_seam_bind_bulk(&rings, map.scratch, AOTX_BULK_STAGE_BYTES);
    if (aotx_catalog_open() != 0) {
        printf("check: the built-in tools did not go in the catalog\n");
        return 1;
    }
    cudaMalloc((void **)&on, sizeof *on);
    /* The head of an import carries the tail of the path of its directory, and that field
     * holds 63 bytes. The check names the directory it was given as the root, so the
     * loader opens the module file whatever the length of the path. */
    aotx_tool_module_root(dir);
    unsigned int entry = aotx_check_import_dir(dir, &held, on);
    if (entry >= AOTX_MODULE_SLOTS) {
        return 1;
    }
    aotx_check_say(held.state == AOTX_CATALOG_INSTALLED, "the manifest of the module:",
                   (unsigned long long)held.state);
    if (held.state != AOTX_CATALOG_INSTALLED) {
        printf("check: the reason is: %s\n", held.reason);
        return 1;
    }
    printf("check: the tool takes %u arguments, side %s, timeout %u, deadline %u\n",
           held.arguments, (held.side == AOTX_CATALOG_SIDE_HOST) ? "host" : "device",
           held.timeout, held.deadline);
    aotx_check_say(held.example_len != 0u, "bytes of the example line:",
                   (unsigned long long)held.example_len);
    if (held.side == AOTX_CATALOG_SIDE_HOST) {
        aotx_check_program(dir, &held, aotx_check_last(dir), &aotx_check_applied,
                           &aotx_check_failed);
    } else {
        aotx_check_device(dir, entry, rows);
    }
    aotx_tool_module_close();
    cudaFree(on);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    printf("check: %u checks applied, %u failed\n", aotx_check_applied, aotx_check_failed);
    return (aotx_check_failed == 0u) ? 0 : 1;
}
