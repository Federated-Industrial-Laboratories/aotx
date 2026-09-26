/* Purpose: Check that shared output keeps completed decode slots until its terminal record commits.
 * Owns: Distinct token, participant, receipt and page-count fixtures without model weights.
 * Launch shape: Real decode, result and record kernels at N=1 and N=64.
 * Lifetime: Token-limit completion through delayed output and final page release. */
#include "model/decode_state.cuh"
#include "model/load.cuh"
#include "shared/bridge.cuh"
#include "shared/internal.cuh"
#include "cognitive/intake.cuh"
#include "cli/prompt.cuh"
#include "text/text.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
static unsigned checks, failures;
static void check(bool good, const char *label)
{ ++checks; if (!good) { ++failures; std::fprintf(stderr, "FAIL %s\n", label); } }
static void cu(cudaError_t status)
{ if (status != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(status)); std::exit(1); } }
struct aotx_completion_row { aotx_seq seq; unsigned owns, pages, asked, shown; };

__global__ void aotx_completion_seed(unsigned first, unsigned count, bool shared)
{
    unsigned slot = threadIdx.x;
    aotx_seqs.slot[slot] = {}; aotx_seq_kept[slot] = aotx_seq_shown[slot] = aotx_seq_asked[slot] = 0;
    aotx_intake.row[slot] = aotx_service.slot[slot] = aotx_shared.slot[slot] = 0;
    aotx_kv.count[slot] = 0; aotx_decode.rows[slot] = 0;
    aotx_shared_execution_slots[slot] = {}; aotx_live_bindings[slot] = {}; aotx_say.slot[slot] = {};
    if (!slot) {
        aotx_sched.held = 0; aotx_seam.replaying = 0; aotx_sched.start_ns = 1;
        aotx_decode.role = AOTX_MODEL_LANGUAGE; aotx_time_tick = 17; aotx_seqs.live = count;
        aotx_kv.made = aotx_kv.served = aotx_kv.refused = 0;
        aotx_model_wrap[AOTX_MODEL_LANGUAGE] = {};
        aotx_model_load.resident[AOTX_MODEL_LANGUAGE].active = 1;
        aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[0] = 77;
    }
    unsigned local = shared ? slot - 1 : slot;
    if (local >= count) return;
    unsigned index = first + local, prompt = 2 + index % 3, pages = 2 + index % 3;
    aotx_seq &seq = aotx_seqs.slot[slot];
    seq.state = AOTX_SEQ_STATE_PREFILL; seq.role = AOTX_MODEL_LANGUAGE;
    seq.prompt = prompt; seq.limit = 1; seq.stop = ~0u; seq.page_limit = pages;
    seq.seed = 1000 + index; seq.opened = 16;
    for (unsigned j = 0; j < prompt; ++j) aotx_seqs.tokens[slot][j] = 100 + index * 8 + j;
    aotx_decode.first[slot] = 0; aotx_decode.rows[slot] = prompt; aotx_decode.place[slot] = slot;
    aotx_decode.token[slot] = index; aotx_decode.draw[slot] = 2000 + index; aotx_model_seen[slot] = prompt;
    aotx_kv.count[slot] = aotx_seq_asked[slot] = pages;
    if (!shared) return;
    aotx_shared_receipt &r = aotx_shared.receipts[index]; r = {};
    r.actor[0] = index + 1; r.id[0] = index + 1; r.key[0] = index + 17;
    r.sequence = index + 100; r.revision = 7; r.phase = AOTX_SHARED_RUNNING;
    r.operation = AOTX_SHARED_INPUT; r.participant = r.space = r.conversation = index;
    r.slot = slot; r.role = AOTX_MODEL_LANGUAGE; r.limit = 1; r.pages = pages;
    r.admission_source = 1; r.saved_admission = 1; r.model_digest[0] = 77;
    aotx_shared.slot[slot] = index + 1;
    auto &person = aotx_shared.participants[index]; person = {};
    person.active = 1; person.id[0] = index + 1; person.next = r.sequence + 1; person.floor = 1;
    auto &space = aotx_shared.spaces[index]; space = {};
    space.active = 1; space.id[0] = index + 100; space.owner[0] = index + 1;
    auto &conversation = aotx_shared.conversations[index]; conversation = {};
    conversation.active = 1; conversation.space = index; conversation.request = index + 1;
    auto &grant = aotx_service.grants[index]; grant = {};
    grant.principal[0] = index + 1; grant.revision = 7; grant.actions = 127;
    grant.models = 1u << AOTX_MODEL_LANGUAGE; grant.tokens = 1; grant.pages = pages;
    aotx_live_bindings[slot].active = 1; aotx_live_bindings[slot].principal[0] = index + 100;
    aotx_shared_execution_slots[slot] = {AOTX_SHARED_DECODE, 0, 1, 1};
}

__global__ void aotx_completion_read(aotx_completion_row *rows)
{
    unsigned slot = threadIdx.x;
    rows[slot] = {aotx_seqs.slot[slot], aotx_shared_owns(slot), aotx_kv.count[slot],
        aotx_seq_asked[slot], aotx_seq_shown[slot]};
}

struct fixture {
    aotx_shared_state state = {}; aotx_service_state service = {}; aotx_seam_state seam = {};
    unsigned char *bytes = nullptr; unsigned long long *offsets = nullptr;
    aotx_completion_row *rows = nullptr;
    fixture(unsigned count) {
        state.enabled = 1; state.participant_capacity = state.space_capacity = state.conversation_capacity = state.receipt_capacity = count;
#define ALLOC(field) cu(cudaMalloc(&state.field, count * sizeof(*state.field))); cu(cudaMemset(state.field, 0, count * sizeof(*state.field)))
        ALLOC(participants); ALLOC(spaces); ALLOC(conversations); ALLOC(receipts);
#undef ALLOC
        service.enabled = 1; service.grant_count = count;
        cu(cudaMalloc(&service.grants, count * sizeof(*service.grants)));
        cu(cudaMemset(service.grants, 0, count * sizeof(*service.grants)));
        cu(cudaMalloc(&rows, AOTX_SLOTS * sizeof(*rows)));
        cu(cudaMalloc(&seam.dev.base, 4096 * AOTX_SLOT_BYTES)); cu(cudaMemset(seam.dev.base, 0, 4096 * AOTX_SLOT_BYTES));
        seam.dev.slot_count = 4096; seam.dev.mask = 4095; seam.apply.state_hash = AOTX_FNV_BASIS;
        std::vector<unsigned char> text(count * 7); std::vector<unsigned long long> at(count + 1);
        for (unsigned i = 0; i < count; ++i) {
            unsigned char token[7] = {'r', (unsigned char)('0' + i / 10), (unsigned char)('0' + i % 10), ':', 'O', 'K', '!'};
            std::memcpy(text.data() + i * 7, token, 7); at[i] = i * 7;
        }
        at[count] = count * 7;
        cu(cudaMalloc(&bytes, text.size())); cu(cudaMemcpy(bytes, text.data(), text.size(), cudaMemcpyHostToDevice));
        cu(cudaMalloc(&offsets, at.size() * sizeof(at[0]))); cu(cudaMemcpy(offsets, at.data(), at.size() * sizeof(at[0]), cudaMemcpyHostToDevice));
        aotx_text_vocab vocab[3] = {}; vocab[0].tokens = count; vocab[0].token_bytes = bytes; vocab[0].token_at = offsets;
        cu(cudaMemcpyToSymbol(aotx_text_vocab_saved, vocab, sizeof(vocab)));
        cu(cudaMemcpyToSymbol(aotx_shared, &state, sizeof(state))); cu(cudaMemcpyToSymbol(aotx_service, &service, sizeof(service)));
        cu(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof(seam)));
    }
    ~fixture() {
        cudaFree(state.participants); cudaFree(state.spaces); cudaFree(state.conversations); cudaFree(state.receipts);
        cudaFree(service.grants); cudaFree(rows); cudaFree(seam.dev.base); cudaFree(bytes); cudaFree(offsets);
    }
    std::vector<aotx_completion_row> observe() {
        std::vector<aotx_completion_row> out(AOTX_SLOTS);
        aotx_completion_read<<<1,AOTX_SLOTS>>>(rows); cu(cudaDeviceSynchronize());
        cu(cudaMemcpy(out.data(), rows, out.size() * sizeof(out[0]), cudaMemcpyDeviceToHost)); return out;
    }
    void held(unsigned first, unsigned count, bool shared) {
        auto current = observe(); unsigned shown = 0;
        for (unsigned i = 0; i < count; ++i) {
            unsigned index = first + i, slot = shared ? i + 1 : i; const auto &r = current[slot]; shown += r.shown;
            check(r.seq.state == AOTX_SEQ_STATE_DONE && r.owns == shared, "completion retains its exact slot owner");
            check(r.pages == 2 + index % 3 && r.asked == r.pages, "completed output keeps its requested pages");
            check(r.seq.prompt == 2 + index % 3 && r.seq.sampled == 1 && r.seq.limit == 1 && r.seq.last == index &&
                r.seq.seed == 1000 + index && r.seq.draw == 2000 + index, "completed batch preserves distinct terminal usage and tokens");
        }
        unsigned live; cu(cudaMemcpyFromSymbol(&live, aotx_seqs, sizeof(live), offsetof(aotx_seq_table, live)));
        check(live == count, "held completions remain in the live sequence count");
        if (shared) check(shown <= 1, "pending output leaves the other result bytes unread");
    }
    void released(unsigned count, bool shared) {
        auto current = observe();
        for (unsigned i = 0; i < count; ++i) {
            const auto &r = current[shared ? i + 1 : i];
            check(r.seq.state == AOTX_SEQ_STATE_FREE && !r.owns && !r.pages && !r.asked, "unowned completion releases its slot and every page");
        }
        auto kv = std::make_unique<aotx_kv_table>(); cu(cudaMemcpyFromSymbol(kv.get(), aotx_kv, sizeof(*kv)));
        check(kv->made == count && !kv->refused, "each completed slot queues exactly one page release");
        std::vector<unsigned> seen(AOTX_SLOTS);
        for (unsigned i = 0; i < kv->made && i < AOTX_KV_QUEUE_MAX; ++i) {
            const auto &entry = kv->queue[i]; check(entry.agent < AOTX_SLOTS && entry.pages == 0, "page queue contains only complete releases");
            if (entry.agent < AOTX_SLOTS) ++seen[entry.agent];
        }
        for (unsigned i = 0; i < count; ++i) check(seen[shared ? i + 1 : i] == 1, "every distinct slot has one release request");
        unsigned live; cu(cudaMemcpyFromSymbol(&live, aotx_seqs, sizeof(live), offsetof(aotx_seq_table, live)));
        check(live == 0, "release clears the complete live sequence count");
    }
};

static void run(unsigned count, bool shared, const char *prefix)
{
    fixture f(count);
    for (unsigned first = 0; first < count;) {
        unsigned n = count - first, width = shared ? AOTX_SLOTS - 1 : AOTX_SLOTS;
        if (n > width) n = width;
        aotx_completion_seed<<<1,AOTX_SLOTS>>>(first, n, shared);
        aotx_decode_commit<<<1,AOTX_SLOTS>>>(17); cu(cudaDeviceSynchronize()); f.held(first, n, shared);
        if (shared) {
            aotx_shared_results<<<1,1>>>(); cu(cudaDeviceSynchronize());
            aotx_shared_state pending; cu(cudaMemcpyFromSymbol(&pending, aotx_shared, sizeof(pending)));
            check(pending.kind == AOTX_SHARED_OUTPUT_RECORD && pending.written == 0, "real result consumption starts a pending output record");
            for (unsigned delay = 0; delay < 5; ++delay) {
                aotx_decode_commit<<<1,AOTX_SLOTS>>>(18 + delay); aotx_shared_results<<<1,1>>>();
                cu(cudaDeviceSynchronize()); f.held(first, n, true);
            }
            std::vector<aotx_shared_receipt> waiting(n);
            cu(cudaMemcpy(waiting.data(), f.state.receipts + first, n * sizeof(waiting[0]), cudaMemcpyDeviceToHost));
            for (const auto &r : waiting) check(r.phase == AOTX_SHARED_RUNNING && !r.output && !r.terminal_source,
                "delayed output cannot publish an early terminal receipt");
            for (unsigned step = 0; step < n * 2 + 4; ++step) {
                aotx_decode_commit<<<1,AOTX_SLOTS>>>(24 + step);
                aotx_shared_results<<<1,1>>>(); aotx_shared_emit<<<1,1>>>();
            }
        }
        for (unsigned delay = 0; delay < 3; ++delay) aotx_decode_commit<<<1,AOTX_SLOTS>>>(count * 2 + 40 + delay);
        cu(cudaDeviceSynchronize()); f.released(n, shared); first += n;
    }
    if (shared) {
        std::vector<aotx_shared_receipt> rows(count); cu(cudaMemcpy(rows.data(), f.state.receipts, count * sizeof(rows[0]), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < count; ++i) {
            unsigned char exact[7] = {'r', (unsigned char)('0' + i / 10), (unsigned char)('0' + i % 10), ':', 'O', 'K', '!'};
            const auto &r = rows[i];
            check(r.phase == AOTX_SHARED_DONE && r.status == 200 && r.finish == 2 && r.slot == AOTX_SLOTS,
                "delayed token-limit completion returns success after output drains");
            check(r.actor[0] == i + 1 && r.sequence == i + 100 && r.prompt == 2 + i % 3 && r.sampled == 1 &&
                r.output == 7 && !std::memcmp(r.result, exact, 7), "each receipt preserves its exact actor, result bytes and usage");
        }
        cu(cudaMemcpyFromSymbol(&f.seam, aotx_seam, sizeof(f.seam)));
        std::vector<unsigned char> records(f.seam.dev.tail * AOTX_SLOT_BYTES);
        cu(cudaMemcpy(records.data(), f.seam.dev.base, records.size(), cudaMemcpyDeviceToHost));
        if (prefix) {
            std::string name = std::string(prefix) + "-" + std::to_string(count) + ".journal";
            FILE *file = fopen(name.c_str(), "wb");
            check(file != nullptr, "journal comparison output opens");
            if (file) {
                check(fwrite(records.data(), 1, records.size(), file) == records.size(), "journal comparison output keeps every byte");
                check(fclose(file) == 0, "journal comparison output closes");
            }
        }
        unsigned outputs = 0, terminals = 0; std::vector<unsigned> ended(count);
        for (unsigned i = 0; i < f.seam.dev.tail; ++i) {
            const auto *h = (const aotx_record_header *)(records.data() + i * AOTX_SLOT_BYTES);
            if (h->type != AOTX_SHARED_RECORD) continue;
            const unsigned char *p = (const unsigned char *)h + AOTX_HEADER_BYTES;
            check(h->cls == AOTX_CLASS_A && aotx_service_get(p + 28, 4) == 0, "exact shared result has an explicit complete class A record");
            unsigned kind = aotx_service_get(p + 4, 4), index = aotx_service_get(p + 32, 4);
            if (kind == AOTX_SHARED_OUTPUT_RECORD) ++outputs;
            if (kind == AOTX_SHARED_COMPLETE_RECORD) {
                ++terminals; check(index < count && aotx_service_get(p + 36, 4) == 200, "recorded completion keeps its successful status");
                if (index < count) { ++ended[index]; check(rows[index].terminal_source == h->seq, "receipt names its exact terminal source record"); }
            }
        }
        check(outputs == count && terminals == count && f.seam.apply.applied_count == 2 * count, "one output and one terminal are recorded per distinct input");
        for (unsigned n : ended) check(n == 1, "every input has exactly one terminal record");
        aotx_shared_state final; cu(cudaMemcpyFromSymbol(&final, aotx_shared, sizeof(final)));
        check(!final.fatal && !final.kind, "all bounded shared result transfers complete");
    }
    std::printf("shared-completion N=%u shared=%u checks=%u failures=%u\n", count, (unsigned)shared, checks, failures);
}
int main(int argc, char **argv)
{
    if (argc > 2) return 2;
    for (unsigned count : {1u, 64u}) for (bool shared : {false, true}) run(count, shared, argc == 2 ? argv[1] : nullptr);
    std::printf("shared-completion total checks=%u failures=%u\n", checks, failures);
    return failures ? 1 : 0;
}
