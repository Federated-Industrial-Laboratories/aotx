/* Purpose: Check advertised argument contracts and exact tool result text.
 * Owns: Catalog fixtures, result buffers, and a bounded device record ring.
 * Launch shape: One thread for each agent at one and the profile slot count.
 * Lifetime: One test process. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "agent/call.cuh"
#include "seam/seam.cuh"
#include "tool/tool_state.cuh"
#include "wrap_fixture.h"
#include "call_schema_json.h"

#define AOTX_SCHEMA_ROLE (AOTX_MODULE_SLOTS - 1u)
#define AOTX_SCHEMA_IMPORT_ROLE (AOTX_MODULE_SLOTS - 2u)
#define AOTX_SCHEMA_FULL_ROLE (AOTX_MODULE_SLOTS - 3u)
#define AOTX_SCHEMA_IMPORT AOTX_CATALOG_BUILT_IN
#define AOTX_SCHEMA_START (AOTX_SAY_BYTES - AOTX_CATALOG_LIST_BYTES)

typedef struct aotx_schema_row {
    unsigned char lists[3][AOTX_SAY_BYTES + 16u];
    unsigned int length[3];
    char calls[6][512];
    char value[64];
    char result[128];
    char note_result[128];
    char error_result[128];
    aotx_tool_call parsed;
    unsigned char frame[AOTX_SAY_BYTES];
    unsigned int frame_length, checks, result_ok;
    unsigned long long note_seq;
} aotx_schema_row;

__device__ static int aotx_schema_equal(const void *left, const void *right,
                                       unsigned int length)
{
    const unsigned char *a = (const unsigned char *)left;
    const unsigned char *b = (const unsigned char *)right;
    for (unsigned int i = 0u; i < length; ++i)
        if (a[i] != b[i]) return 0;
    return 1;
}

/* Parse each fixture manifest through the same reader as a built-in or imported entry. */
__device__ static int aotx_schema_manifest(unsigned int entry, const char *text,
                                           unsigned int kind)
{
    unsigned int length = 0u, figure = 0u;
    while (text[length]) ++length;
    aotx_catalog_run run;
    if (aotx_catalog_take_run(length, &run)) return 0;
    memcpy(aotx_catalog_arena + run.at, text, length);
    aotx_catalog_entry *row = &aotx_catalog.entry[entry];
    memset(row, 0, sizeof *row);
    row->manifest = run;
    row->kind = kind;
    if (aotx_catalog_manifest_read(row, run.at, length, kind, &figure)) return 0;
    row->state = AOTX_CATALOG_INSTALLED;
    return 1;
}

__global__ void aotx_schema_setup(const char *conductor, unsigned char *records,
                                  unsigned int *okay)
{
    if (threadIdx.x != 0u) return;
    *okay = aotx_schema_manifest(AOTX_SCHEMA_ROLE, conductor, AOTX_MODULE_ROLE);
    *okay &= aotx_schema_manifest(AOTX_SCHEMA_IMPORT,
        "kind: tool\nname: external_note\nside: host\narguments: provenance\n"
        "authorise: never\nprogram: note.sh\ndescription: Read the supplied source.\n",
        AOTX_MODULE_TOOL);
    aotx_catalog_mask_clear(aotx_catalog.entry[AOTX_SCHEMA_IMPORT_ROLE].role.tools);
    aotx_catalog_mask_set(aotx_catalog.entry[AOTX_SCHEMA_IMPORT_ROLE].role.tools,
                            AOTX_SCHEMA_IMPORT);
    aotx_catalog_mask_clear(aotx_catalog.entry[AOTX_SCHEMA_FULL_ROLE].role.tools);
    for (unsigned int i = 0u; i <= AOTX_SCHEMA_IMPORT; ++i)
        aotx_catalog_mask_set(aotx_catalog.entry[AOTX_SCHEMA_FULL_ROLE].role.tools, i);
    aotx_tool_embed.ready = 1u;
    aotx_seam.dev.base = records;
    aotx_seam.dev.mask = 4u * AOTX_SLOTS - 1u;
    aotx_seam.dev.slot_count = 4u * AOTX_SLOTS;
}

__global__ void aotx_schema_prepare(aotx_schema_row *rows, unsigned int count)
{
    unsigned int slot = threadIdx.x;
    memset(&aotx_requests.slot[slot], 0, sizeof aotx_requests.slot[slot]);
    aotx_tool_done[slot] = 0u;
    aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_NONE;
    if (slot == 0u) {
        aotx_seam.dev.tail = 0ull;
        aotx_embed_notes.count = aotx_embed_notes.width = 0u;
        aotx_tool_embed.width = 1u;
    }
    if (slot >= count) return;
    aotx_schema_row *row = &rows[slot];
    for (unsigned int c = 0u; c < 6u; ++c) {
        unsigned int length = 0u;
        while (row->calls[c][length]) ++length;
        int parsed = aotx_tool_parse((const unsigned char *)row->calls[c], length, &row->parsed);
        row->checks += parsed == (c == 4u ? 3 : 1);
        if (c < 4u) row->checks += row->parsed.provenance == AOTX_PROV_COMPUTED + c;
        if (c != 4u) continue;
        row->checks += row->parsed.error == AOTX_TOOL_CALL_PROVENANCE;
        unsigned int id = aotx_tool_error_request(slot, &row->parsed, 1ull);
        const aotx_request *request = &aotx_requests.slot[slot];
        unsigned int start = 0u, size = 0u, value_length = 0u;
        while (row->value[value_length]) ++value_length;
        row->checks += id != 0u && request->status == AOTX_TOOL_ERROR && aotx_tool_done[slot] == 1u;
        row->checks += aotx_tool_argument_of(request->arg, request->arg_len,
            "provenance", 10u, &start, &size) && size == value_length
            && aotx_schema_equal(request->arg + start, row->value, value_length);
        if (request->result_len < sizeof row->error_result) {
            memcpy(row->error_result, request->result, request->result_len);
            row->error_result[request->result_len] = '\0';
        }
        aotx_requests.slot[slot].request = 0u;
    }
    unsigned int length = 0u;
    while (row->calls[3][length]) ++length;
    aotx_tool_parse((const unsigned char *)row->calls[3], length, &row->parsed);
    row->checks += aotx_tool_request(slot, &row->parsed, 0u, 2ull) != 0u;
    aotx_tool_embed.place[slot] = slot;
    aotx_tool_embed.vector[slot] = 1.0f;
    aotx_tool_embed.state[slot] = AOTX_TOOL_EMBED_RUN;
}

__global__ void aotx_schema_render(aotx_schema_row *rows, unsigned int count)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    aotx_schema_row *row = &rows[slot];
    row->length[0] = aotx_catalog_tool_list(row->lists[0], 0u, AOTX_SCHEMA_ROLE);
    row->length[1] = aotx_catalog_tool_list(row->lists[1], 0u, AOTX_SCHEMA_IMPORT_ROLE);
    row->length[2] = aotx_catalog_tool_list(row->lists[2], AOTX_SCHEMA_START, AOTX_SCHEMA_FULL_ROLE);
    unsigned int length = 0u;
    while (row->result[length]) ++length;
    row->frame_length = aotx_call_result(row->frame, 0u,
        (const unsigned char *)row->result, 0u, length, sizeof row->result);
    const aotx_request *request = &aotx_requests.slot[slot];
    row->result_ok = aotx_tool_done[slot] && request->status == AOTX_TOOL_OK;
    if (request->result_len < sizeof row->note_result) {
        memcpy(row->note_result, request->result, request->result_len);
        row->note_result[request->result_len] = '\0';
    }
    for (unsigned int n = 0u; n < aotx_embed_notes.count; ++n) {
        if (aotx_embed_notes.len[n] != row->parsed.arg_len
            || !aotx_schema_equal(aotx_embed_notes.text[n], row->parsed.arg, row->parsed.arg_len)) continue;
        row->note_seq = aotx_embed_notes.seq[n];
        const unsigned char *bytes = aotx_seam_body_of(row->note_seq);
        const aotx_record_header *header = (const aotx_record_header *)(bytes - AOTX_HEADER_BYTES);
        const aotx_bus_body *body = (const aotx_bus_body *)bytes;
        row->result_ok &= header->seq == row->note_seq && header->type == AOTX_REC_BUS
            && header->writer == AOTX_WRITER_AGENT_BASE + slot
            && body->kind == AOTX_BUS_FINDING && body->provenance == AOTX_PROV_TESTIMONY
            && body->text_len == row->parsed.arg_len
            && aotx_schema_equal(body->text, row->parsed.arg, row->parsed.arg_len)
            && aotx_embed_notes.vector[n][0] == 1.0f;
    }
}

static unsigned int applied, failed;
static void check(int okay, const char *text, unsigned int kind, unsigned int slot)
{
    ++applied;
    if (okay) return;
    ++failed;
    printf("call schema: kind %u slot %u: %s\n", kind, slot, text);
}

static void run(unsigned int kind, unsigned int count)
{
    aotx_schema_row *host = (aotx_schema_row *)calloc(count, sizeof *host), *device = NULL;
    if (host == NULL) exit(2);
    aotx_call_format format;
    aotx_call_format_make(kind, &format);
    aotx_test_call_upload(kind);
    for (unsigned int slot = 0u; slot < count; ++slot) {
        aotx_schema_row *row = &host[slot];
        memset(row->lists, 0x5a, sizeof row->lists);
        snprintf(row->value, sizeof row->value, "operator_source_%u", slot);
        snprintf(row->result, sizeof row->result, "saved %u\n\"ok\"\\", slot);
        const char *values[] = {"computed", "fetched", "recalled", "testimony", row->value, row->value};
        for (unsigned int c = 0u; c < 6u; ++c) {
            const char *name = c == 5u ? "external_note" : "memory_write";
            char second[128] = "";
            if (kind == AOTX_CALL_QWEN_XML) {
                if (c != 5u) snprintf(second, sizeof second,
                    "<parameter=text>\nnote value %u\n</parameter>\n", slot);
                snprintf(row->calls[c], sizeof row->calls[c],
                    "<tool_call>\n<function=%s>\n<parameter=provenance>\n%s\n</parameter>\n%s"
                    "</function>\n</tool_call>", name, values[c], second);
            } else {
                if (c != 5u) snprintf(second, sizeof second, ",\"text\":\"note value %u\"", slot);
                snprintf(row->calls[c], sizeof row->calls[c],
                    "%s{\"name\":\"%s\",\"%s\":{\"provenance\":\"%s\"%s}}%s",
                    kind == AOTX_CALL_HERMES ? "<tool_call>" : "", name,
                    kind == AOTX_CALL_HERMES ? "arguments" : "parameters", values[c], second,
                    kind == AOTX_CALL_HERMES ? "</tool_call>" : "");
            }
        }
    }
    aotx_check_runtime(cudaMalloc(&device, count * sizeof *device), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device, host, count * sizeof *host, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_schema_prepare<<<1, AOTX_SLOTS>>>(device, count);
    aotx_tool_step<<<1, AOTX_SLOTS>>>(3ull);
    aotx_schema_render<<<1, AOTX_SLOTS>>>(device, count);
    aotx_check_runtime(cudaMemcpy(host, device, count * sizeof *host, cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int slot = 0u; slot < count; ++slot) {
        aotx_schema_row *row = &host[slot];
        if (row->checks != 14u || !row->result_ok || row->note_seq == 0ull)
            printf("call schema: kind %u slot %u checks=%u result=%u seq=%llu text=%s\n",
                kind, slot, row->checks, row->result_ok, row->note_seq, row->note_result);
        unsigned int mask = 0u, tools = 0u;
        check(aotx_schema_list(row->lists[0], row->length[0], &format, &mask, &tools)
              && mask == 15u && tools == 4u, "four conductor schemas are complete", kind, slot);
        check(aotx_schema_list(row->lists[1], row->length[1], &format, &mask, &tools)
              && mask == 16u && tools == 1u, "imported provenance has no built-in restriction", kind, slot);
        check(row->length[2] >= AOTX_SCHEMA_START
              && aotx_schema_list(row->lists[2] + AOTX_SCHEMA_START,
                 row->length[2] - AOTX_SCHEMA_START, &format, &mask, &tools)
              && tools > 0u && tools < AOTX_CATALOG_BUILT_IN + 1u,
              "full list cuts only complete schemas and keeps its closing text", kind, slot);
        unsigned int guard = 1u;
        for (unsigned int b = 0u; b < AOTX_SCHEMA_START; ++b)
            guard &= row->lists[2][b] == 0x5au;
        for (unsigned int list = 0u; list < 3u; ++list)
            for (unsigned int b = AOTX_SAY_BYTES; b < AOTX_SAY_BYTES + 16u; ++b)
                guard &= row->lists[list][b] == 0x5au;
        check(guard, "list writes stay inside the prompt", kind, slot);
        check(row->checks == 14u, "provenance calls retain their contract and invalid value", kind, slot);
        check(!strcmp(row->error_result,
            "provenance must be computed, fetched, recalled, or testimony; no note was saved"),
            "invalid provenance gives the complete refusal", kind, slot);
        char expected[512];
        check(row->result_ok && row->note_seq != 0ull
              && !strcmp(row->note_result, "the note is in memory"),
              "memory result is independent of the stored record sequence", kind, slot);
        if (kind == AOTX_CALL_LLAMA_JSON) {
            snprintf(expected, sizeof expected,
                "<|start_header_id|>ipython<|end_header_id|>\n\n"
                "\"saved %u\\u000a\\\"ok\\\"\\\\\"<|eot_id|>", slot);
        } else {
            /* Exact local templates put one newline between the user role and the result tag. */
            snprintf(expected, sizeof expected,
                "<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n", row->result);
        }
        check(row->frame_length == strlen(expected)
              && !memcmp(row->frame, expected, strlen(expected)), "result framing matches the template", kind, slot);
    }
    printf("call schema: kind %u N=%u conductor list %u/%u bytes\n",
           kind, count, host[0].length[0], AOTX_CATALOG_LIST_BYTES);
    aotx_check_runtime(cudaFree(device), "cudaFree");
    free(host);
}

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    FILE *file = fopen(argv[1], "rb");
    if (file == NULL) return 2;
    char text[4096] = {};
    size_t bytes = fread(text, 1u, sizeof text - 1u, file);
    int read_ok = bytes > 0u && !ferror(file) && feof(file);
    fclose(file);
    if (!read_ok) return 2;
    aotx_check_runtime(cudaSetDevice(0), "cudaSetDevice");
    if (aotx_catalog_open()) return 2;
    aotx_test_wrap_open();
    char *manifest = NULL;
    unsigned char *records = NULL;
    unsigned int *okay = NULL, setup = 0u;
    aotx_check_runtime(cudaMalloc(&manifest, bytes + 1u), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&records, 4u * AOTX_SLOTS * AOTX_SLOT_BYTES), "cudaMalloc");
    aotx_check_runtime(cudaMalloc(&okay, sizeof *okay), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(manifest, text, bytes + 1u, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_schema_setup<<<1, 1>>>(manifest, records, okay);
    aotx_check_runtime(cudaMemcpy(&setup, okay, sizeof setup, cudaMemcpyDeviceToHost), "cudaMemcpy");
    if (!setup) { fprintf(stderr, "call schema: fixture manifest is invalid\n"); return 2; }
    for (unsigned int kind = AOTX_CALL_HERMES; kind < AOTX_CALL_FORMAT_KINDS; ++kind) {
        run(kind, 1u);
        run(kind, AOTX_SLOTS);
    }
    aotx_check_runtime(cudaFree(manifest), "cudaFree");
    aotx_check_runtime(cudaFree(records), "cudaFree");
    aotx_check_runtime(cudaFree(okay), "cudaFree");
    printf("call schema: %u checks, %u failed\n", applied, failed);
    return failed != 0u;
}
