/* Purpose: Check selected call forms, packed values and refusal without call residue.
 * Owns: Distinct reply batches and their expected values.
 * Launch shape: One device thread for each reply, at one and 64 replies.
 * Lifetime: One run of tool_test.cu. */
#ifndef AOTX_TEST_TOOL_FORMAT_H
#define AOTX_TEST_TOOL_FORMAT_H

#include "wrap_fixture.h"

static unsigned int aotx_tool_format_text(unsigned int kind, unsigned int i,
    unsigned int mode, char *out, unsigned int max, char *value, char *source)
{
    static const char *const provenance[] = {"computed", "fetched", "recalled", "testimony"};
    if (mode == 4u) snprintf(source, 64u, "guessed-%u", i);
    else snprintf(source, 64u, "%s", provenance[(i + mode) % 4u]);
    snprintf(value, 128u, mode == 2u
        ? "\n note %u {\"x\":1} </parameter> <tool_call> </function> \n"
        : " note %u {\"x\":1} <tool_call> </function> ", i);
    char encoded[256];
    unsigned int at = 0u;
    for (unsigned int b = 0u; value[b] != '\0'; ++b) {
        if (value[b] == '\n') {
            encoded[at++] = '\\';
            encoded[at++] = 'n';
            continue;
        }
        if (value[b] == '"' || value[b] == '\\') encoded[at++] = '\\';
        encoded[at++] = value[b];
    }
    encoded[at] = '\0';
    const char *key = kind == AOTX_CALL_LLAMA_JSON ? "parameters" : "arguments";
    if (kind == AOTX_CALL_QWEN_XML) {
        const char *first = mode % 2u == 0u ? "text" : "provenance";
        const char *second = mode % 2u == 0u ? "provenance" : "text";
        if (mode == 3u) {
            return (unsigned int)snprintf(out, max,
                "<tool_call><function=memory_write><parameter=provenance>%s</parameter>"
                "<parameter=text>%s</parameter></function></tool_call>", source, value);
        }
        return (unsigned int)snprintf(out, max,
            "<tool_call>\n<function=memory_write>\n<parameter=%s>\n%s\n</parameter>\n"
            "<parameter=%s>\n%s\n</parameter>\n</function>\n</tool_call>",
            first, mode % 2u == 0u ? value : source,
            second, mode % 2u == 0u ? source : value);
    }
    char args[384];
    if (mode % 2u == 0u) {
        snprintf(args, sizeof args, "{\"text\":\"%s\",\"provenance\":\"%s\"}", encoded, source);
    } else {
        snprintf(args, sizeof args, "{\"provenance\":\"%s\",\"text\":\"%s\"}", source, encoded);
    }
    const char *head = kind == AOTX_CALL_HERMES ? "<tool_call>"
                     : mode == 5u ? "<|python_tag|>" : "";
    const char *tail = kind == AOTX_CALL_HERMES ? "</tool_call>" : "";
    if ((mode & 2u) != 0u) {
        return (unsigned int)snprintf(out, max, "%s{\"%s\":%s,\"name\":\"memory_write\"}%s",
                                      head, key, args, tail);
    }
    return (unsigned int)snprintf(out, max, "%s{\"name\":\"memory_write\",\"%s\":%s}%s",
                                  head, key, args, tail);
}

static unsigned int aotx_tool_format_clear(const aotx_tool_call *call)
{
    unsigned int wrong = call->entry != AOTX_MODULE_SLOTS || call->tool != AOTX_TOOL_NONE
        || call->key != AOTX_CATALOG_ARGS || call->provenance != 0u || call->over != 0u
        || call->error != 0u || call->values != 0u || call->arg_len != 0u || call->pack_len != 0u;
    for (unsigned int k = 0u; k < AOTX_CATALOG_ARGS; ++k) {
        wrong += call->at[k] != 0u || call->length[k] != 0u;
    }
    return wrong;
}

static void aotx_tool_format_scan(unsigned int kind, unsigned int count,
    const aotx_tool_test_batch *batch, int expected, aotx_tool_call *back,
    unsigned int *wrong)
{
    aotx_test_call_upload(kind);
    unsigned char *text = (unsigned char *)aotx_tool_test_take(sizeof batch->text);
    unsigned int *start = (unsigned int *)aotx_tool_test_take(sizeof batch->start);
    unsigned int *length = (unsigned int *)aotx_tool_test_take(sizeof batch->length);
    aotx_tool_call *calls = (aotx_tool_call *)aotx_tool_test_take(count * sizeof *calls);
    int *found = (int *)aotx_tool_test_take(count * sizeof(int));
    int marks[AOTX_TOOL_CASE_GOOD];
    aotx_check_runtime(cudaMemcpy(text, batch->text, sizeof batch->text, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(start, batch->start, sizeof batch->start, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(length, batch->length, sizeof batch->length, cudaMemcpyHostToDevice), "cudaMemcpy");
    aotx_check_runtime(cudaMemset(calls, 0xa5, count * sizeof *calls), "cudaMemset");
    aotx_tool_scan<<<1, 64>>>(text, start, length, count, calls, found);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_check_runtime(cudaMemcpy(back, calls, count * sizeof *calls, cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(marks, found, count * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy");
    for (unsigned int i = 0u; i < count; ++i) {
        *wrong += marks[i] != expected;
        if (expected == 0) *wrong += aotx_tool_format_clear(&back[i]);
    }
    cudaFree(text); cudaFree(start); cudaFree(length); cudaFree(calls); cudaFree(found);
}

static void aotx_tool_format_cases(unsigned int count, unsigned int *applied,
                                    unsigned int *failed)
{
    aotx_tool_test_batch *batch = (aotx_tool_test_batch *)calloc(1, sizeof *batch);
    aotx_tool_call *back = (aotx_tool_call *)calloc(count, sizeof *back);
    char value[128], source[64], text[AOTX_TOOL_CASE_BYTES];
    unsigned int wrong = 0u;
    for (unsigned int kind = AOTX_CALL_HERMES; kind <= AOTX_CALL_QWEN_XML; ++kind) {
        for (unsigned int mode = 0u; mode < 6u; ++mode) {
            for (unsigned int i = 0u; i < count; ++i) {
                batch->start[i] = i * AOTX_TOOL_CASE_BYTES;
                batch->length[i] = aotx_tool_format_text(kind, i, mode,
                    batch->text + batch->start[i], AOTX_TOOL_CASE_BYTES, value, source);
            }
            aotx_tool_format_scan(kind, count, batch, mode == 4u ? 3 : 1, back, &wrong);
            for (unsigned int i = 0u; i < count; ++i) {
                aotx_tool_format_text(kind, i, mode, text, sizeof text, value, source);
                const aotx_tool_call *call = &back[i];
                wrong += call->tool != AOTX_TOOL_MEMORY_WRITE || call->values != 2u
                    || call->arg_len != strlen(value) || memcmp(call->arg, value, strlen(value))
                    || call->error != (mode == 4u ? AOTX_TOOL_CALL_PROVENANCE : 0u);
                static const unsigned int provenance[] = {
                    AOTX_PROV_COMPUTED, AOTX_PROV_FETCHED,
                    AOTX_PROV_RECALLED, AOTX_PROV_TESTIMONY
                };
                wrong += call->provenance != (mode == 4u ? 0u : provenance[(i + mode) % 4u]);
                unsigned int matched = 0u;
                for (unsigned int k = 0u; k < AOTX_CATALOG_ARGS; ++k) {
                    if (call->length[k] == strlen(source)
                        && call->at[k] + call->length[k] <= call->pack_len
                        && memcmp(call->pack + call->at[k], source, strlen(source)) == 0) matched++;
                }
                wrong += matched != 1u;
            }
            for (unsigned int other = AOTX_CALL_NONE; other <= AOTX_CALL_QWEN_XML; ++other) {
                if (other != kind) aotx_tool_format_scan(other, count, batch, 0, back, &wrong);
            }
            if (mode == 4u) {
                for (unsigned int i = 0u; i < count; ++i) batch->length[i]--;
                aotx_tool_format_scan(kind, count, batch, 0, back, &wrong);
            }
        }
        for (unsigned int over = 0u; over < 2u; ++over) {
            char path[AOTX_TOOL_ARG_BYTES + 1u];
            unsigned int bytes = AOTX_TOOL_ARG_BYTES - 6u + over;
            for (unsigned int i = 0u; i < count; ++i) {
                memset(path, 'a' + i % 26u, bytes);
                path[bytes - 1u] = (char)('0' + i / 26u);
                path[bytes] = '\0';
                batch->start[i] = i * AOTX_TOOL_CASE_BYTES;
                const char *shape = kind == AOTX_CALL_QWEN_XML
                    ? "<tool_call><function=fs_read><parameter=path>\n%s\n</parameter></function></tool_call>"
                    : kind == AOTX_CALL_HERMES
                    ? "<tool_call>{\"name\":\"fs_read\",\"arguments\":{\"path\":\"%s\"}}</tool_call>"
                    : "{\"name\":\"fs_read\",\"parameters\":{\"path\":\"%s\"}}";
                batch->length[i] = (unsigned int)snprintf(batch->text + batch->start[i],
                    AOTX_TOOL_CASE_BYTES, shape, path);
            }
            aotx_tool_format_scan(kind, count, batch, over != 0u ? 2 : 1, back, &wrong);
            for (unsigned int i = 0u; i < count; ++i) {
                wrong += back[i].over != over || back[i].error != 0u
                    || back[i].tool != AOTX_TOOL_FS_READ
                    || back[i].arg_len != (over != 0u ? 0u : bytes)
                    || back[i].pack_len != (over != 0u ? 0u : bytes)
                    || back[i].values != (over != 0u ? 0u : 1u);
                if (over != 0u) {
                    for (unsigned int k = 0u; k < AOTX_CATALOG_ARGS; ++k) {
                        wrong += back[i].at[k] != 0u || back[i].length[k] != 0u;
                    }
                }
            }
        }
    }
    static const char *const bad[] = {
        "{\"name\":\"fs_read\",\"arguments\":{\"path\":\"x\"}}",
        "#tool_call {\"name\":\"fs_read\",\"parameters\":{\"path\":\"x\"}}",
        "prefix {\"name\":\"fs_read\",\"parameters\":{\"path\":\"x\"}}",
        "{\"name\":\"fs_read\",\"parameters\":{\"path\":\"x\"}} suffix",
        "{\"name\":\"fs_read\",\"parameters\":{\"path\":\"x\"}} {}",
        "{\"name\":\"fs_read\",\"parameters\":{\"path\":\"x\",\"path\":\"y\"}}",
        "{\"name\":\"fs_read\",\"parameters\":{\"unknown\":\"x\"}}",
        "{\"name\":\"memory_write\",\"parameters\":{\"text\":\"x\"}}",
        "{\"name\":\"fs_read\",\"name\":\"fs_read\",\"parameters\":{\"path\":\"x\"}}",
        "{\"parameters\":{\"path\":\"x\",},\"name\":\"fs_read\"}",
        "<|python_tag|>print(1)",
        "<|python_tag|><|python_tag|>{\"name\":\"fs_read\",\"parameters\":{\"path\":\"x\"}}",
        "<tool_call><function=fs_read><parameter=path>x</parameter><parameter=path>y</parameter></function></tool_call>",
        "<tool_call><function=fs_read><parameter=unknown>x</parameter></function></tool_call>",
        "<tool_call><function=memory_write><parameter=text>x</parameter></function></tool_call>",
        "<tool_call><function=fs_read><parameter=path>x</function></tool_call>",
        "<tool_call><function=fs_read><parameter=path>x</parameter></function></tool_call> suffix"
    };
    for (unsigned int c = 0u; c < sizeof bad / sizeof bad[0]; ++c) {
        for (unsigned int i = 0u; i < count; ++i) {
            batch->start[i] = i * AOTX_TOOL_CASE_BYTES;
            batch->length[i] = (unsigned int)snprintf(batch->text + batch->start[i],
                                                     AOTX_TOOL_CASE_BYTES, "%s", bad[c]);
        }
        aotx_tool_format_scan(c < 12u ? AOTX_CALL_LLAMA_JSON : AOTX_CALL_QWEN_XML,
                              count, batch, 0, back, &wrong);
    }
    aotx_tool_service_check(wrong, "selected call forms", count, applied, failed);
    aotx_test_call_upload(AOTX_CALL_HERMES);
    free(back); free(batch);
}

#endif
