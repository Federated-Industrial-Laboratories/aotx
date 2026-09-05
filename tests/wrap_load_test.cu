/* Purpose: Check loaded wraps and the rejection of changed spans and end tokens.
 * Owns: One model load and temporary mutation fixtures.
 * Launch shape: The load check uses tokenizer batches and a short device prefill.
 * Lifetime: One check run. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "boot/boot.cuh"
#include "boot/check.h"
#include "mem/mem.cuh"
#include "model/roles.h"
#include "model/wrap_check.cuh"
#include "model/forward.cuh"
#include "model/conduct.cuh"
#include "model/layout_host.h"
extern "C" {
#include "disk/modelfile/manifest.h"
}

static unsigned int checks, failed;
static void check(int good, const char *name)
{
    ++checks;
    if (!good) ++failed;
    printf("wrap load: %s %s\n", good ? "pass" : "FAIL", name);
}

__global__ void aotx_wrap_logits(float *logits, unsigned int count, unsigned int best,
                                 aotx_wrap_check_work *work)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) logits[i] = i == best ? 10.0f : -1.0f;
    if (i == 0u) { work->order_ok = 1u; work->ends_ok = 1u; aotx_model_faults = 0u; }
}

__global__ void aotx_wrap_mass_seed(void)
{
    unsigned int slot = blockIdx.x, page = blockIdx.y * blockDim.x + threadIdx.x;
    if (slot < AOTX_SLOTS && page < AOTX_KV_PAGES_EACH)
        aotx_page_mass[slot][page] = (float)(slot * AOTX_KV_PAGES_EACH + page + 1u);
}

static void aotx_wrap_unknown_replace(const char *models, unsigned int role)
{
    char dir[] = "/tmp/aotx-wrap-replace-XXXXXX";
    if (mkdtemp(dir) == NULL) { check(0, "replacement fixture opens"); return; }
    char path[256];
    snprintf(path, sizeof path, "%s/unknown.gguf", dir);
    FILE *file = fopen(path, "wb");
    if (file == NULL) { check(0, "replacement fixture writes"); rmdir(dir); return; }
    const uint32_t prefix[] = { AOTX_GGUF_MAGIC, AOTX_GGUF_VERSION };
    const uint64_t counts[] = { 0u, 1u };
    const char key[] = "tokenizer.chat_template", text[] = "unknown";
    uint64_t key_bytes = sizeof key - 1u, text_bytes = sizeof text - 1u;
    uint32_t type = AOTX_GGUF_STRING;
    int bad = fwrite(prefix, sizeof prefix, 1u, file) != 1u
           || fwrite(counts, sizeof counts, 1u, file) != 1u
           || fwrite(&key_bytes, sizeof key_bytes, 1u, file) != 1u
           || fwrite(key, 1u, key_bytes, file) != key_bytes
           || fwrite(&type, sizeof type, 1u, file) != 1u
           || fwrite(&text_bytes, sizeof text_bytes, 1u, file) != 1u
           || fwrite(text, 1u, text_bytes, file) != text_bytes;
    while (!bad && ftell(file) % AOTX_GGUF_ALIGN != 0) bad = fputc(0, file) == EOF;
    bad |= fclose(file) != 0;
    if (bad) { check(0, "replacement fixture is complete"); unlink(path); rmdir(dir); return; }
    aotx_manifest_entry entries[AOTX_MODEL_FILES_MAX];
    int count = aotx_manifest_read(models, entries, AOTX_MODEL_FILES_MAX);
    if (count <= 0 || count >= AOTX_MODEL_FILES_MAX) {
        check(0, "replacement manifest has room"); unlink(path); rmdir(dir); return;
    }
    aotx_manifest_entry *entry = &entries[count];
    memset(entry, 0, sizeof *entry);
    snprintf(entry->path, sizeof entry->path, "%s", path);
    snprintf(entry->name, sizeof entry->name, "unknown-wrap");
    snprintf(entry->role, sizeof entry->role, "%s", aotx_role_name[role]);
    snprintf(entry->source, sizeof entry->source, "local");
    snprintf(entry->revision, sizeof entry->revision, "1");
    snprintf(entry->license, sizeof entry->license, "test");
    entry->probe_numerator = 2u; entry->probe_denominator = 3u;
    unsigned char buffer[256];
    if (aotx_sha256_file(path, entry->sha256, &entry->bytes, buffer, sizeof buffer)) {
        check(0, "replacement digest is complete"); unlink(path); rmdir(dir); return;
    }
    aotx_model_load_state state;
    aotx_check_runtime(cudaMemcpyFromSymbol(&state, aotx_model_load, sizeof state),
                       "cudaMemcpyFromSymbol");
    state.files = (unsigned int)count + 1u;
    unsigned long long held = aotx_mem_weights_held(), cursor = held, bytes = 0ull;
    unsigned int placed = 0u, left = 0u;
    int result = aotx_model_layout_replace(models, entries, (unsigned int)count + 1u,
                                          &state, (unsigned int)count,
                                          &cursor, &placed, &left, &bytes);
    check(result != 0, "replacement refuses an unknown wrap before placement");
    check(held != 0ull && aotx_mem_weights_held() == held && cursor == held,
          "wrap refusal leaves resident weight allocations unchanged");
    unlink(path); rmdir(dir);
}

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: wrap_load_test <models> <role>\n"); return 2; }
    unsigned int role = aotx_role_of(argv[2]);
    if (role >= AOTX_MODEL_ROLES || !aotx_model_is_language(role)) return 2;
    aotx_check_runtime(cudaFree(0), "cudaFree");
    aotx_mem_map map;
    if (aotx_mem_reserve(&map) || aotx_boot_models(argv[1], argv[2], NULL)) return 1;
    aotx_wrap original;
    aotx_model_desc desc;
    aotx_check_runtime(cudaMemcpyFromSymbol(&original, aotx_model_wrap, sizeof original,
                                            role * sizeof original), "cudaMemcpyFromSymbol");
    aotx_check_runtime(cudaMemcpyFromSymbol(&desc, aotx_model, sizeof desc,
                                            role * sizeof desc), "cudaMemcpyFromSymbol");
    check(original.usable == 1u, "loaded file passes the complete check");
    check(desc.probe_layer < desc.layers, "derived probe layer is inside the model");
    aotx_wrap changed = original;
    unsigned int first = AOTX_WRAP_USER_HEAD, second = AOTX_WRAP_ASSISTANT_HEAD;
    uint16_t offset = changed.offset[first]; uint8_t length = changed.length[first];
    changed.offset[first] = changed.offset[second]; changed.length[first] = changed.length[second];
    changed.offset[second] = offset; changed.length[second] = length;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &changed, sizeof changed,
                                          role * sizeof changed), "cudaMemcpyToSymbol");
    check(aotx_model_wrap_check(role, &changed, "swapped-heads") != 0,
          "swapped table heads fail the load check");
    changed = original;
    changed.end_ids[0] = desc.vocab;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &changed, sizeof changed,
                                          role * sizeof changed), "cudaMemcpyToSymbol");
    check(aotx_model_wrap_check(role, &changed, "invalid-end-id") != 0,
          "end id outside the vocabulary fails the load check");
    changed = original;
    changed.end_ids[0] = 0u;
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &changed, sizeof changed,
                                          role * sizeof changed), "cudaMemcpyToSymbol");
    check(aotx_model_wrap_check(role, &original, "changed-uploaded-end") != 0,
          "uploaded end ids must match the store table");
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &original, sizeof original,
                                          role * sizeof original), "cudaMemcpyToSymbol");
    aotx_wrap_check_work *work;
    float *logits;
    aotx_check_runtime(cudaMalloc(&work, sizeof *work), "cudaMalloc");
    aotx_check_runtime(cudaMemset(work, 0, sizeof *work), "cudaMemset");
    aotx_check_runtime(cudaMalloc(&logits, desc.vocab * sizeof(float)), "cudaMalloc");
    aotx_wrap_logits<<<(desc.vocab + 255u) / 256u, 256>>>(logits, desc.vocab,
                                                        original.end_ids[0], work);
    aotx_wrap_check_argmax<<<1, 256>>>(role, logits, work);
    unsigned int passed;
    aotx_check_runtime(cudaMemcpy(&passed, &work->prefill_ok, sizeof passed,
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    check(passed == 0u, "an end-token argmax fails the prefill check");
    aotx_wrap loaded;
    aotx_check_runtime(cudaMemcpyFromSymbol(&loaded, aotx_model_wrap, sizeof loaded,
                                            role * sizeof loaded), "cudaMemcpyFromSymbol");
    aotx_model_desc retained;
    aotx_check_runtime(cudaMemcpyFromSymbol(&retained, aotx_model, sizeof retained,
                                            role * sizeof retained), "cudaMemcpyFromSymbol");
    check(loaded.usable == 0u && memcmp(&retained, &desc, sizeof desc) == 0,
          "failed check disables say and keeps the descriptor");
    cudaFree(logits); cudaFree(work);
    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_wrap, &original, sizeof original,
                                          role * sizeof original), "cudaMemcpyToSymbol");
    aotx_wrap_mass_seed<<<dim3(AOTX_SLOTS, (AOTX_KV_PAGES_EACH + 127u) / 128u), 128>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check(aotx_model_wrap_check(role, &original, "restored-table") == 0,
          "restored table passes the complete load check");
    size_t cells = (size_t)AOTX_SLOTS * AOTX_KV_PAGES_EACH;
    float *mass = (float *)malloc(cells * sizeof(float));
    if (mass == NULL) return 1;
    aotx_check_runtime(cudaMemcpyFromSymbol(mass, aotx_page_mass, cells * sizeof(float)),
                       "cudaMemcpyFromSymbol");
    unsigned int unchanged = 1u;
    for (size_t i = 0u; i < cells; ++i) unchanged &= mass[i] == (float)(i + 1u);
    check(unchanged, "private prefill preserves every slot's attention mass");
    free(mass);
    aotx_wrap_unknown_replace(argv[1], role);
    printf("wrap load: %u checks, %u failed\n", checks, failed);
    aotx_boot_models_release();
    aotx_mem_release(&map);
    return failed ? 1 : 0;
}
