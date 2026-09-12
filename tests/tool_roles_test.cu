/* Purpose: Check reply tool parsing and continuation with distinct language owners.
 * Owns: Completed reply fixtures, opposite call protocols and exact argument checks.
 * Launch shape: N=1 and N=64 through the production agent consumer.
 * Lifetime: One test process; no tool action is executed. */
#include "agent/prompt.cuh"
#include "wrap_fixture.h"
#include <string>
#include <vector>

static unsigned checks, failures;
static void check(bool good, const char *name)
{
    ++checks; if (!good) { ++failures; fprintf(stderr, "FAIL %s\n", name); }
}
static void cu(cudaError_t rc)
{
    if (rc != cudaSuccess) { fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(rc)); exit(2); }
}
struct aotx_tool_role_row {
    unsigned role, form, stage, tool, entry, bytes, resumed_role, prompt_bytes, next_role;
    char reply[512], arg[128], result[64], prompt[AOTX_SAY_BYTES];
};
__global__ void aotx_tool_roles_seed(unsigned n, const unsigned char *bytes, const unsigned long long *at)
{
    unsigned slot = threadIdx.x; if (slot >= AOTX_SLOTS) return;
    aotx_agents.agent[slot] = {}; aotx_agent_gear[slot] = {};
    aotx_seqs.slot[slot] = {}; aotx_say.slot[slot] = {}; aotx_transcript[slot] = {};
    if (!slot) {
        aotx_model[AOTX_MODEL_LANGUAGE].layers = aotx_model[AOTX_MODEL_LANGUAGE_AUDIO].layers = 1;
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE].shape, 1, 1, 32);
        aotx_kvl_make(&aotx_model_space[AOTX_MODEL_LANGUAGE_AUDIO].shape, 1, 1, 32);
        aotx_text_vocab vocab = {}; vocab.tokens = n; vocab.token_bytes = bytes; vocab.token_at = at;
        aotx_text_vocab_saved[0] = aotx_text_vocab_saved[2] = vocab;
    }
    if (slot >= n) return;
    unsigned role = n > 1 && slot % 2 == 0 ? AOTX_MODEL_LANGUAGE : AOTX_MODEL_LANGUAGE_AUDIO;
    aotx_prompt_roles[slot] = AOTX_MODEL_LANGUAGE;
    aotx_agents.agent[slot].state = AOTX_AGENT_STATE_RUN;
    aotx_agents.agent[slot].task = ~0u;
    aotx_agents.agent[slot].role = AOTX_MODULE_SLOTS - 1;
    aotx_transcript[slot].pages = AOTX_KV_PAGES_EACH;
    aotx_seqs.slot[slot].state = AOTX_SEQ_STATE_DONE; aotx_seqs.slot[slot].role = role;
    aotx_seqs.slot[slot].sampled = 1; aotx_seqs.slot[slot].limit = 2; aotx_seqs.slot[slot].stop = ~0u;
    aotx_seqs.tokens[slot][0] = slot;
}
__global__ void aotx_tool_roles_read(aotx_tool_role_row *out, unsigned n)
{
    unsigned slot = threadIdx.x; if (slot >= n) return;
    auto &r = out[slot]; auto &gear = aotx_agent_gear[slot];
    r.role = aotx_seqs.slot[slot].role; r.form = aotx_call_format_active(r.role)->kind;
    r.stage = aotx_agents.agent[slot].state; r.tool = gear.call.tool;
    r.entry = gear.call.entry; r.bytes = gear.reply_len;
    for (unsigned j = 0; j < gear.reply_len && j < 511; ++j) r.reply[j] = gear.reply[j];
    for (unsigned j = 0; j < gear.call.arg_len && j < 127; ++j) r.arg[j] = gear.call.arg[j];
    if (r.tool == AOTX_TOOL_NONE || r.entry >= AOTX_MODULE_SLOTS) return;
    const unsigned char query[] = "Read the named file.";
    unsigned result_bytes = 0; while (r.result[result_bytes]) ++result_bytes;
    r.prompt_bytes = aotx_agent_prompt(slot, 0, query, sizeof(query)-1, 0, 0, 0, r.result, result_bytes);
    r.resumed_role = aotx_prompt_role(slot);
    if (r.prompt_bytes < AOTX_SAY_BYTES)
        for (unsigned j = 0; j < r.prompt_bytes; ++j) r.prompt[j] = aotx_say.prompt[slot][j];
    aotx_say.slot[slot].wanted = 0;
    aotx_agent_prompt(slot, 0, query, sizeof(query)-1, 0, 0, 0, 0, 0);
    r.next_role = aotx_prompt_role(slot);
}
static void exercise(unsigned n, unsigned mode)
{
    unsigned kinds[2] = {mode == 1 ? AOTX_CALL_NONE : AOTX_CALL_QWEN_XML,
        mode == 0 ? AOTX_CALL_NONE : mode == 1 ? AOTX_CALL_QWEN_XML : AOTX_CALL_HERMES};
    aotx_wrap wrap = aotx_test_wrap_table();
    for (unsigned i = 0; i < 2; ++i) {
        unsigned role = i ? AOTX_MODEL_LANGUAGE_AUDIO : AOTX_MODEL_LANGUAGE;
        aotx_call_format format = {}; if (aotx_call_format_make(kinds[i], &format)) exit(2);
        cu(cudaMemcpyToSymbol(aotx_model_call_format, &format, sizeof(format), role*sizeof(format)));
        cu(cudaMemcpyToSymbol(aotx_model_wrap, &wrap, sizeof(wrap), role*sizeof(wrap)));
    }
    std::vector<std::string> texts, arguments; std::string joined;
    std::vector<unsigned long long> offsets;
    std::vector<aotx_tool_role_row> out(n);
    for (unsigned i = 0; i < n; ++i) {
        bool audio = n == 1 || i % 2 != 0;
        arguments.push_back("source-file-" + std::to_string(mode) + "-" + std::to_string(i));
        texts.push_back(audio && mode == 2 ? "<tool_call>{\"name\":\"fs_read\",\"arguments\":{\"path\":\"" + arguments.back() + "\"}}</tool_call>"
            : "<tool_call>\n<function=fs_read>\n<parameter=path>\n" + arguments.back() + "\n</parameter>\n</function>\n</tool_call>");
        offsets.push_back(joined.size()); joined += texts.back();
        snprintf(out[i].result, sizeof(out[i].result), "file result %u %u", mode, i);
    }
    offsets.push_back(joined.size());
    unsigned char *bytes; unsigned long long *at; aotx_tool_role_row *rows;
    cu(cudaMalloc(&bytes, joined.size())); cu(cudaMemcpy(bytes, joined.data(), joined.size(), cudaMemcpyHostToDevice));
    cu(cudaMalloc(&at, offsets.size()*sizeof(*at))); cu(cudaMemcpy(at, offsets.data(), offsets.size()*sizeof(*at), cudaMemcpyHostToDevice));
    cu(cudaMalloc(&rows, n*sizeof(*rows))); cu(cudaMemcpy(rows, out.data(), n*sizeof(*rows), cudaMemcpyHostToDevice));
    aotx_tool_roles_seed<<<1,64>>>(n, bytes, at); cu(cudaDeviceSynchronize());
    aotx_agent_step<<<1,64>>>(123); cu(cudaDeviceSynchronize());
    aotx_tool_roles_read<<<1,64>>>(rows, n); cu(cudaDeviceSynchronize());
    cu(cudaMemcpy(out.data(), rows, n*sizeof(*rows), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < n; ++i) {
        const auto &r = out[i]; bool audio = n == 1 || i % 2 != 0;
        bool wanted = kinds[audio] != AOTX_CALL_NONE;
        check(r.bytes == texts[i].size() && !memcmp(r.reply, texts[i].data(), r.bytes), "completed reply bytes are exact");
        check(r.stage == AOTX_AGENT_STATE_POST, "completed sequence reaches the reply consumer");
        bool parsed = r.tool != AOTX_TOOL_NONE && r.entry < AOTX_MODULE_SLOTS;
        check(parsed == wanted, "only the sequence owner's protocol parses the reply");
        if (!wanted || !parsed) continue;
        check(r.arg == arguments[i], "parsed argument belongs to its sequence");
        check(r.resumed_role == r.role && r.prompt_bytes && r.prompt_bytes < AOTX_SAY_BYTES,
            "tool continuation keeps the completed sequence owner");
        check(strstr(r.prompt, r.result) != 0, "the distinct tool result reaches its continuation");
        check(r.next_role == AOTX_MODEL_LANGUAGE, "a new text input selects the default model");
    }
    cu(cudaFree(rows)); cu(cudaFree(at)); cu(cudaFree(bytes));
    printf("tool roles N=%u mode=%u checks=%u failures=%u\n", n, mode, checks, failures);
}
int main(void)
{
    cu(cudaSetDevice(0)); if (aotx_catalog_open()) return 2;
    for (unsigned n : {1u, 64u}) for (unsigned mode = 0; mode < 3; ++mode) exercise(n, mode);
    return failures ? 1 : 0;
}
