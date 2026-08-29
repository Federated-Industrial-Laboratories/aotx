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
#include "catalog/check_host.h"
#include "settings/keys.h"

extern "C" {
#include "disk/settings/settings.h"
}
#include "tool/module_host.h"
#include "seam/seam.cuh"

unsigned int aotx_check_applied;
unsigned int aotx_check_failed;


void aotx_check_say(int ok, const char *what, unsigned long long figure)
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
 * back. The head carries the digest the caller gives, as the head of the feeder does. The
 * return is the entry, or AOTX_MODULE_SLOTS. */
unsigned int aotx_check_import_dir(const char *dir, aotx_check_entry *out,
                                          aotx_check_entry *on,
                                          const unsigned char *digest, unsigned int number)
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
    unsigned char *on_digest = NULL;
    cudaMalloc((void **)&on_bytes, length + 1u);
    cudaMemcpy(on_bytes, bytes, length, cudaMemcpyHostToDevice);
    free(bytes);
    if (digest != NULL) {
        cudaMalloc((void **)&on_digest, 32u);
        cudaMemcpy(on_digest, digest, 32u, cudaMemcpyHostToDevice);
    }
    char *on_name = aotx_check_text(aotx_check_last(dir), AOTX_IMPORT_NAME_BYTES);
    char *on_path = aotx_check_text(dir, AOTX_IMPORT_PATH_BYTES);
    aotx_check_import<<<1, 1>>>(on_bytes, length, AOTX_MODULE_TOOL, on_name, on_path,
                                on_digest, number);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaFree(on_bytes);
    cudaFree(on_name);
    cudaFree(on_path);
    cudaFree(on_digest);

    /* The entry of the built-in tools stands before this one, so the import takes the
     * first free entry after them. An import of the same name takes that entry again. */
    unsigned int at = AOTX_CATALOG_BUILT_IN;
    aotx_check_report<<<1, 1>>>(at, on);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(out, on, sizeof *out, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    return at;
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
     * holds 63 bytes. The root of the loader is the directory that holds the module
     * directories, which is the one above the directory given. The loader then opens the
     * module file whatever the length of the path. */
    char root[1024];
    const char *tail = strrchr(dir, '/');
    snprintf(root, sizeof root, "%.*s", (int)((tail != NULL) ? (tail - dir) : 1), 
             (tail != NULL) ? dir : ".");
    aotx_tool_module_root(root);
    unsigned int entry = aotx_check_import_dir(dir, &held, on, NULL, 1u);
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
        aotx_check_device(dir, entry, rows, &held, on);
    }
    aotx_tool_module_close();
    cudaFree(on);
    aotx_seam_close(&rings);
    aotx_mem_release(&map);
    cuDevicePrimaryCtxRelease(device);
    printf("check: %u checks applied, %u failed\n", aotx_check_applied, aotx_check_failed);
    return (aotx_check_failed == 0u) ? 0 : 1;
}
