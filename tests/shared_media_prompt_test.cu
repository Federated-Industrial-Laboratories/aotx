/* Purpose: Keep recalled shared media links literal and expand only the current explicit input.
 * Owns: Distinct scoped source descriptors, prompt bytes and exact expected reference lists.
 * Launch shape: Real media preparation and expansion at N=1 and N=64.
 * Lifetime: One prompt batch, with ordinary history and invalid current-source controls. */
#include "media/runtime.cuh"
#include "media/prompt.cuh"
#include "shared/state.cuh"
#include "cli/prompt.cuh"
#include "cognitive/intake.cuh"
#include "model/decode.cuh"
#include "model/load.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
static unsigned checks, failures;
static constexpr unsigned text_bytes = 384;
static void check(bool value, const char *label)
{ ++checks; if (!value) { ++failures; std::fprintf(stderr, "FAIL %s\n", label); } }
static void cu(cudaError_t status)
{ if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); } }
struct aotx_media_case {
    unsigned length, turn;
    unsigned char context[32], current[32], text[text_bytes];
};
struct aotx_media_result {
    unsigned stage, count, extra, turn_extra, raw_length, raw_turn, length, turn, expanded, feature_rows, wrong;
    aotx_media_reference refs[2];
    unsigned char text[text_bytes];
};
struct aotx_media_admission { unsigned status, refs, object, model; unsigned long long generation; };

__global__ void aotx_shared_media_seed(const aotx_media_case *cases, unsigned first, unsigned count,
                                      unsigned total, unsigned mode, bool shared, bool audio)
{
    unsigned slot = threadIdx.x;
    aotx_say.slot[slot] = {}; aotx_media_prompts[slot] = {}; aotx_seqs.slot[slot] = {};
    aotx_live_bindings[slot] = {}; aotx_intake.row[slot] = 0; aotx_shared.slot[slot] = aotx_service.slot[slot] = 0;
    aotx_prompt_roles[slot] = audio ? AOTX_MODEL_LANGUAGE_AUDIO : AOTX_MODEL_LANGUAGE;
    unsigned local = shared ? slot - 1 : slot;
    if (local >= count) return;
    unsigned index = first + local;
    const auto &input = cases[index];
    auto &binding = aotx_live_bindings[slot]; binding.active = 1;
    binding.principal[0] = shared ? 200 : index + 1; binding.principal[1] = shared ? index + 1 : 0;
    binding.room[0] = 55; binding.room[1] = 1 + index / 2;
    for (unsigned source = 0; source < 2; ++source) {
        auto &object = aotx_media.objects[source * total + index]; object = {};
        object.scope = AOTX_MEDIA_PRIVATE; object.phase = AOTX_MEDIA_READY;
        object.principal[0] = index + 1; object.room[0] = 55; object.room[1] = 1 + index / 2;
        object.transfer[0] = source + 1; object.transfer[1] = index + 1;
        object.generation = (source ? 1000 : 100) + index; object.worker = ~0u;
        bool family = source ? audio : mode == 2 ? !audio : audio;
        if (source && mode == 5) family = !family;
        object.format = family ? AOTX_AUDIO_WAV : AOTX_IMAGE_JPEG;
        object.rows = source ? 2 + index % 3 : 1 + index % 4;
        object.span = object.lines = object.rows; object.columns = 1; object.feature = (source * total + index) * 4;
        for (unsigned j = 0; j < 32; ++j) object.digest[j] = source ? input.current[j] : input.context[j];
        if ((!source && (mode == 1 || mode == 7)) || (source && mode == 4)) object.phase = AOTX_MEDIA_REFUSED;
        if ((!source && mode == 3) || (source && mode == 6)) object.principal[0] ^= 128;
    }
    if (shared) {
        auto &receipt = aotx_shared.receipts[index]; receipt = {};
        receipt.actor[0] = index + 1; receipt.sequence = index + 100; receipt.phase = AOTX_SHARED_RUNNING;
        receipt.slot = slot; receipt.role = aotx_prompt_roles[slot]; receipt.media_count = mode == 7 ? 0 : 1;
        receipt.media[0] = {total + index, 1000 + index}; aotx_shared.slot[slot] = index + 1;
    }
    auto &say = aotx_say.slot[slot]; say.wanted = 1; say.length = input.length; say.turn_at = input.turn;
    for (unsigned j = 0; j < input.length; ++j) aotx_say.prompt[slot][j] = input.text[j];
}

__global__ void aotx_shared_media_observe(aotx_media_result *out, unsigned first, unsigned count,
                                         unsigned total, unsigned mode, bool shared, bool audio)
{
    unsigned local = threadIdx.x;
    if (local >= count) return;
    unsigned slot = shared ? local + 1 : local, index = first + local;
    auto &result = out[index]; result = {};
    const auto &m = aotx_media_prompts[slot]; const auto &s = aotx_say.slot[slot];
    result.stage = m.stage; result.count = m.count; result.extra = m.extra; result.turn_extra = m.turn_extra;
    result.raw_length = m.raw_length; result.raw_turn = m.raw_turn; result.length = s.length; result.turn = s.turn_at;
    for (unsigned j = 0; j < 2 && j < m.count; ++j) result.refs[j] = m.reference[j];
    for (unsigned j = 0; j < text_bytes; ++j) result.text[j] = j < s.length ? aotx_say.prompt[slot][j] : 0;
    unsigned start = audio ? AOTX_MEDIA_AUDIO_START : AOTX_MEDIA_VISION_START;
    unsigned pad = audio ? AOTX_MEDIA_AUDIO_PAD : AOTX_MEDIA_IMAGE_PAD;
    unsigned end = audio ? AOTX_MEDIA_AUDIO_END : AOTX_MEDIA_VISION_END;
    unsigned *ids = aotx_say_id + slot * AOTX_SAY_TOKENS, used = 0;
    ids[used++] = 10;
    if (!shared && mode == 0) { ids[used++] = start; ids[used++] = pad; ids[used++] = end; }
    ids[used++] = 11;
    if (mode != 7) { ids[used++] = start; ids[used++] = pad; ids[used++] = end; }
    ids[used++] = 12; result.expanded = aotx_media_expand(slot, used);
    if (!m.count) return;
    for (unsigned j = 0; j < result.expanded; ++j) {
        const auto &row = aotx_media_input[slot][j];
        if (!row.feature) continue;
        unsigned object = !shared && mode == 0 && result.feature_rows < 1 + index % 4 ? index : total + index;
        unsigned prior = object == index ? 0 : !shared && mode == 0 ? 1 + index % 4 : 0;
        unsigned position = result.feature_rows - prior, width = audio ? 4096 : 1024;
        const float *features = audio ? aotx_audio_runtime.features : aotx_media.features;
        result.wrong += row.feature != features + (object * 4ull + position) * width ||
            row.width != width || row.generation != (object == index ? 100 : 1000) + index;
        ++result.feature_rows;
    }
}

__global__ void aotx_shared_media_admission(aotx_media_admission *out, unsigned count, unsigned mode, bool audio)
{
    unsigned index = threadIdx.x, role = audio ? AOTX_MODEL_LANGUAGE_AUDIO : AOTX_MODEL_LANGUAGE;
    if (!index) {
        aotx_live.ready = 1; aotx_live.fatal = 0;
        aotx_model_load.resident[role].active = 1; aotx_model_load.resident[role].body.digest[0] = 77 + role;
        aotx_model_wrap[role].usable = 1;
    }
    __syncthreads();
    if (index >= count) return;
    auto &r = aotx_shared.receipts[index]; r = {}; r.actor[0] = index + 1;
    aotx_service_grant grant = {}; grant.principal[0] = index + 1;
    grant.models = 1u << role; grant.tokens = 8; grant.pages = 4;
    unsigned char *text = r.command + AOTX_SHARED_COMMAND_HEAD;
    const char *prefixes[] = {"<|vision_", "<|image_pad|>", "<|video_pad|>", "<|audio_", "<|AUDIO|>"};
    unsigned length = 0;
    if (mode != 2) {
        text[length++] = 'r'; text[length++] = '0' + index / 10; text[length++] = '0' + index % 10;
        if (mode >= 3) {
            for (unsigned j = 0; j < index % 5; ++j) text[length++] = ' ';
            const char *p = prefixes[mode - 3];
            for (unsigned j = 0; p[j]; ++j) text[length++] = (unsigned char)p[j];
        } else {
            const char *p = " ordinary text";
            for (unsigned j = 0; p[j]; ++j) text[length++] = (unsigned char)p[j];
        }
    }
    aotx_media_put(r.command + 104, role, 4); aotx_media_put(r.command + 108, 8, 4);
    aotx_media_put(r.command + 112, 4, 4); aotx_media_put(r.command + 124, 0x3f800000, 4);
    aotx_media_put(r.command + 136, length, 4);
    if (mode == 1 || mode == 2) {
        aotx_media_put(r.command + 140, 1, 4); aotx_media_put(text + length, audio ? 2 : 1, 4);
        for (unsigned j = 0; j < 32; ++j) text[length + 8 + j] = aotx_media.objects[count + index].digest[j];
    }
    unsigned status = aotx_shared_input_check(&r, &grant);
    out[index] = {status, r.media_count, r.media[0].object, r.model_digest[0], r.media[0].generation};
}

static std::string link(bool audio, const unsigned char *digest)
{
    const char *hex = "0123456789abcdef";
    std::string text = audio ? "[audio:" : "[image:";
    for (unsigned j = 0; j < 32; ++j) { text += hex[digest[j] >> 4]; text += hex[digest[j] & 15]; }
    return text + ']';
}
static std::string marker(bool audio, unsigned number)
{
    return audio ? "Audio " + std::to_string(number) + ": <|audio_bos|><|AUDIO|><|audio_eos|>\n" :
        "<|vision_start|><|image_pad|><|vision_end|>";
}

static void run(unsigned count, bool shared, bool audio)
{
    aotx_shared_state state = {}; state.enabled = shared; state.receipt_capacity = count;
    cu(cudaMalloc(&state.receipts, count * sizeof(*state.receipts))); cu(cudaMemset(state.receipts, 0, count * sizeof(*state.receipts)));
    aotx_service_state service = {}; cu(cudaMemcpyToSymbol(aotx_service, &service, sizeof(service)));
    cu(cudaMemcpyToSymbol(aotx_shared, &state, sizeof(state)));
    aotx_media_state media = {}; media.enabled = media.image_enabled = 1; media.role = AOTX_MODEL_LANGUAGE;
    media.profile.objects = 2 * count; media.profile.feature_rows = 8 * count;
    aotx_audio_runtime_state sound = {}; sound.enabled = 1; sound.role = AOTX_MODEL_LANGUAGE_AUDIO; sound.profile.feature_rows = 8 * count;
    cu(cudaMalloc(&media.objects, media.profile.objects * sizeof(*media.objects)));
    cu(cudaMemset(media.objects, 0, media.profile.objects * sizeof(*media.objects)));
    cu(cudaMalloc(&media.features, 8ull * count * 1024 * sizeof(float)));
    cu(cudaMalloc(&sound.features, 8ull * count * 4096 * sizeof(float)));
    cu(cudaMemcpyToSymbol(aotx_media, &media, sizeof(media))); cu(cudaMemcpyToSymbol(aotx_audio_runtime, &sound, sizeof(sound)));
    aotx_media_case *inputs; aotx_media_result *results;
    cu(cudaMalloc(&inputs, count * sizeof(*inputs))); cu(cudaMalloc(&results, count * sizeof(*results)));
    for (unsigned mode = 0; mode < 8; ++mode) {
        std::vector<aotx_media_case> cases(count); std::vector<std::string> expected(count); std::vector<unsigned> turns(count);
        for (unsigned i = 0; i < count; ++i) {
            auto &row = cases[i];
            for (unsigned j = 0; j < 32; ++j) { row.context[j] = i * 3 + j * 7; row.current[j] = i * 3 + j * 7; }
            row.context[0] = 1; row.current[0] = 2; row.context[1] = row.current[1] = i + 1;
            std::string head = "Memory " + std::to_string(i) + ": ", middle = "\n", prefix = mode % 2 ? "Current " + std::to_string(i) + ": " : "";
            std::string context = head + link(mode == 2 ? !audio : audio, row.context) + middle;
            std::string current = prefix + (mode == 7 ? "Plain text" : link(audio, row.current)) + " end " + std::to_string(i);
            std::string original = context + current; row.turn = context.size(); row.length = original.size();
            check(original.size() < text_bytes, "distinct context and input fit the bounded prompt fixture");
            std::memcpy(row.text, original.data(), original.size());
            std::string prior = shared ? context : head + marker(audio, 1) + middle;
            expected[i] = prior + prefix + (mode == 7 ? "Plain text" : marker(audio, shared ? 1 : 2)) + " end " + std::to_string(i);
            turns[i] = prior.size();
        }
        cu(cudaMemcpy(inputs, cases.data(), count * sizeof(cases[0]), cudaMemcpyHostToDevice));
        for (unsigned first = 0; first < count;) {
            unsigned n = count - first, width = shared ? AOTX_SLOTS - 1 : AOTX_SLOTS;
            if (n > width) n = width;
            aotx_shared_media_seed<<<1,AOTX_SLOTS>>>(inputs, first, n, count, mode, shared, audio);
            aotx_media_prepare<<<1,AOTX_SLOTS>>>();
            aotx_shared_media_observe<<<1,AOTX_SLOTS>>>(results, first, n, count, mode, shared, audio); first += n;
        }
        cu(cudaDeviceSynchronize()); std::vector<aotx_media_result> out(count);
        cu(cudaMemcpy(out.data(), results, count * sizeof(out[0]), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < count; ++i) {
            const auto &r = out[i]; bool valid = shared ? mode < 4 || mode == 7 : mode == 0;
            if (!valid) { check(r.stage == 2 && !r.expanded, "invalid current sources and ordinary invalid history still refuse the prompt"); continue; }
            unsigned refs = mode == 7 ? 0 : shared ? 1 : 2, rows = mode == 7 ? 0 : 2 + i % 3;
            unsigned extra = rows ? rows - 1 : 0;
            if (!shared) { rows += 1 + i % 4; extra += i % 4; }
            check(r.stage == 1 && r.count == refs && r.extra == extra && r.turn_extra == (mode == 7 ? 0 : 1 + i % 3),
                "only permitted input references contribute feature and current-turn counts");
            check(r.length == expected[i].size() && !std::memcmp(r.text, expected[i].data(), expected[i].size()) && r.turn == turns[i],
                "prompt bytes and current-turn boundary preserve literal shared memory links");
            check(r.raw_length == cases[i].length && r.raw_turn == cases[i].turn, "raw prompt coordinates remain exact");
            if (refs) {
                unsigned at = shared ? 0 : 1;
                check(r.refs[at].object == count + i && r.refs[at].generation == 1000 + i, "current media keeps its exact object and generation");
                if (!shared) check(r.refs[0].object == i && r.refs[0].generation == 100 + i, "ordinary history retains its earlier feature reference");
            }
            check(r.expanded == 3 + refs * 3 + extra && r.feature_rows == rows && !r.wrong,
                "real expansion uses only the exact ordered feature addresses");
        }
        std::printf("shared-media-prompt N=%u shared=%u audio=%u mode=%u checks=%u failures=%u\n",
            count, (unsigned)shared, (unsigned)audio, mode, checks, failures);
    }
    if (shared) {
        aotx_media_admission *admission; cu(cudaMalloc(&admission, count * sizeof(*admission)));
        for (unsigned mode = 0; mode < 8; ++mode) {
            aotx_shared_media_admission<<<1,AOTX_SLOTS>>>(admission, count, mode, audio); cu(cudaDeviceSynchronize());
            std::vector<aotx_media_admission> out(count);
            cu(cudaMemcpy(out.data(), admission, count * sizeof(out[0]), cudaMemcpyDeviceToHost));
            for (unsigned i = 0; i < count; ++i) {
                check(out[i].status == (mode < 3 ? 200u : 400u), "reserved marker text fails at real shared input admission");
                if (mode < 3) check(out[i].model == 77 + (audio ? AOTX_MODEL_LANGUAGE_AUDIO : AOTX_MODEL_LANGUAGE) &&
                    out[i].refs == (mode != 0), "plain text and typed media retain the frozen current model");
                if (mode == 1 || mode == 2) check(out[i].object == count + i && out[i].generation == 1000 + i,
                    "typed media admission binds only the exact current private source");
            }
        }
        cudaFree(admission);
    }
    cudaFree(inputs); cudaFree(results); cudaFree(state.receipts); cudaFree(media.objects); cudaFree(media.features); cudaFree(sound.features);
}
int main()
{
    for (unsigned count : {1u, 64u}) for (bool shared : {false, true}) for (bool audio : {false, true}) run(count, shared, audio);
    std::printf("shared-media-prompt total checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
