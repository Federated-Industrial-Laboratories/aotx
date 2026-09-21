/* Purpose: Verify ordered statement extraction and complete classification admission.
 * Owns: Exact spans, excluded text, empty outputs and two-call replay controls.
 * Launch shape: N=1 and N=64 through the device parser, grammar and live state.
 * Lifetime: One test process with supplied responses; no semantic accuracy claim. */
#include "intake_fixture.h"
#include "source_fixture.h"
#include "cognitive/intake_token.cuh"
#include <chrono>

static std::string aotx_stage_first(unsigned i) { return "Person" + std::to_string(i) + " will read."; }
static std::string aotx_stage_second(unsigned i) { return "Their colleague is Partner" + std::to_string(i) + "."; }
static std::string aotx_stage_item(unsigned kind, const std::string &quote) {
    return "[" + std::to_string(kind) + ",\"" + quote + "\",0]";
}
static std::string aotx_stage_statements(unsigned i) {
    return "[" + aotx_stage_item(3, aotx_stage_first(i)) + "," + aotx_stage_item(3, aotx_stage_second(i)) + "]";
}
static std::string aotx_stage_label(unsigned kind, const std::string &quote) {
    return "[\"" + quote + "\",\"" + (kind == 3 ? "statement" : "request") + "\"]";
}
static std::string aotx_stage_extracted(unsigned i) {
    return "[" + aotx_stage_label(3, aotx_stage_first(i)) + "," + aotx_stage_label(0, "Is the door open?") + "," +
        aotx_stage_label(3, aotx_stage_second(i)) + "," + aotx_stage_label(0, "Reply briefly.") + "]";
}
static std::string aotx_stage_mixed(unsigned i) {
    return "[" + aotx_stage_label(3, aotx_stage_first(i)) + "," + aotx_stage_label(0, "Is the door open?") + "," +
        aotx_stage_label(3, aotx_stage_second(i)) + "," + aotx_stage_label(0, "Reply briefly.") + "]";
}
static std::string aotx_stage_rejected(unsigned i) {
    return "[" + aotx_stage_label(0, aotx_stage_first(i)) + "," + aotx_stage_label(0, "Is the door open?") + "," +
        aotx_stage_label(0, aotx_stage_second(i)) + "," + aotx_stage_label(0, "Reply briefly.") + "]";
}
static aotx_bytes aotx_stage_query(unsigned n, bool whitespace = false) {
    auto query = aotx_intake_query(n, 0, 1);
    for (unsigned i = 0; i < n; ++i) {
        auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW; aotx_source_query(q, 8000 + i);
        auto source = aotx_stage_first(i) + " Is the door open? " + aotx_stage_second(i) + " Reply briefly.";
        if (whitespace) source = " \t\n ";
        memset(q + 4640, 0, AOTX_RECALL_TEXT); memcpy(q + 4640, source.data(), source.size());
        aotx_put(q + 148, source.size(), 4);
    }
    return query;
}
static void aotx_stage_prompts(unsigned n) {
    for (bool unknown : {false, true}) {
        aotx_intake_device d(n); aotx_fixture empty;
        d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
        auto query = aotx_stage_query(n);
        if (unknown) for (unsigned i = 0; i < n; ++i)
            memset(query.data() + 128 + i * AOTX_LIVE_QUERY_ROW + AOTX_RECALL_ACTOR, 0, 16);
        d.process(aotx_live_parts(query, 4, d.next_id++), false, false);
        auto *say = new aotx_say_state;
        AOTX_CUDA(cudaMemcpyFromSymbol(say, aotx_say, sizeof(*say)));
        for (unsigned i = 0; i < n; ++i) {
            auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            std::string source((const char *)q + 4640, aotx_get(q + 148, 4));
            std::string expected = "<|im_start|>user\nSource actor: " +
                (unknown ? std::string("unknown") : aotx_source_hex(8000 + i)) +
                "\n<source>\n1: \"" + aotx_stage_first(i) + "\"\n2: \"Is the door open?\"\n3: \"" +
                aotx_stage_second(i) + "\"\n4: \"Reply briefly.\"\n</source>\nFor each whole span: information is statement; a direction or question to the reader is request. Return the array.\n<|im_end|>\n";
            std::string actual((const char *)say->prompt[i], std::min(say->slot[i].length, AOTX_SAY_BYTES));
            auto at = actual.find("<|im_start|>user\n");
            aotx_check(say->slot[i].wanted && at != std::string::npos && actual.substr(at, expected.size()) == expected,
                "the first user frame preserves exact source, actor and classification reminder bytes");
            aotx_check(actual.find("Prior assertions") == std::string::npos,
                "the first call has no prior assertion table");
        }
        delete say;
    }
}
__global__ void aotx_stage_parse(unsigned n, unsigned phase, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; r->phase = phase; r->prefix = {}; r->status = r->bytes = r->count = 0;
    while (r->bytes < AOTX_INTAKE_REPLY && aotx_intake_fixture_first[i][r->bytes]) {
        r->reply[r->bytes] = aotx_intake_fixture_first[i][r->bytes]; ++r->bytes;
    }
    unsigned bytes = r->bytes; r->bytes = 0;
    aotx_seqs.slot[i].role = AOTX_MODEL_LANGUAGE; aotx_seqs.slot[i].stop = UINT32_MAX;
    out[2 * n + i] = aotx_intake_allows(i, i);
    r->bytes = bytes;
    out[i] = aotx_intake_advance(i, r->reply, r->bytes);
    out[n + i] = aotx_intake_parse(i);
}
static void aotx_stage_check(unsigned n, unsigned phase, const std::vector<std::string> &replies, bool expected) {
    aotx_intake_fixture_first_upload(replies);
    std::string encoded; unsigned char *raw; unsigned long long *offset;
    AOTX_CUDA(cudaMallocManaged(&offset, (n + 1) * sizeof(*offset)));
    for (unsigned i = 0; i < n; ++i) {
        offset[i] = encoded.size();
        for (unsigned char byte : replies[i]) {
            if (byte <= 32) { unsigned point = 256 + byte; encoded += char(192 | (point >> 6)); encoded += char(128 | (point & 63)); }
            else encoded += char(byte);
        }
    }
    offset[n] = encoded.size(); AOTX_CUDA(cudaMallocManaged(&raw, encoded.size())); memcpy(raw, encoded.data(), encoded.size());
    aotx_text_vocab vocab = {}; vocab.tokens = n; vocab.token_at = offset; vocab.token_bytes = raw;
    AOTX_CUDA(cudaMemcpyToSymbol(aotx_text_vocab_table, &vocab, sizeof(vocab)));
    unsigned *out; AOTX_CUDA(cudaMallocManaged(&out, 3 * n * sizeof(*out)));
    aotx_stage_parse<<<1,64>>>(n, phase, out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) {
        if (!!out[i] != expected || !out[n + i] != expected)
            fprintf(stderr, "stage %u row %u grammar %u parser %u expected %u\n", phase, i, out[i], out[n + i], expected);
        aotx_check(!!out[i] == expected, "live prefix admission enforces the declared statement boundary");
        aotx_check(!!out[2 * n + i] == expected, "fused raw-token admission uses each advancing mandatory statement position");
        aotx_check(!out[n + i] == expected, "independent complete parsing enforces the same boundary");
    }
    aotx_text_vocab cleared = {}; AOTX_CUDA(cudaMemcpyToSymbol(aotx_text_vocab_table, &cleared, sizeof(cleared)));
    cudaFree(out); cudaFree(raw); cudaFree(offset);
}
static void aotx_stage_spans(unsigned n) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    d.process(aotx_live_parts(aotx_stage_query(n), 4, d.next_id++), false, false);
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    std::vector<std::string> accepted;
    for (unsigned i = 0; i < n; ++i) accepted.push_back(aotx_stage_extracted(i));
    for (unsigned mode = 0; mode < 20; ++mode) {
        std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i) {
            auto first = aotx_stage_first(i), second = aotx_stage_second(i);
            std::string reply = accepted[i];
            if (mode == 1) reply = "[" + aotx_stage_label(3, second) + "," + aotx_stage_label(3, first) + "]";
            if (mode == 2) reply = "[" + aotx_stage_label(3, first) + "," + aotx_stage_label(3, "will read") + "]";
            if (mode == 3) reply = "[" + aotx_stage_item(1, first) + "]";
            if (mode == 4) reply = "[" + aotx_stage_label(3, "absent text") + "]";
            if (mode == 5) reply = "[         ]";
            if (mode == 6) reply = "[]";
            if (mode == 7) reply = "[" + aotx_stage_label(0, "Is the door open?") + "]";
            if (mode == 8) reply = aotx_stage_mixed(i);
            if (mode == 9) reply = "[[\"Is the door open?\",\"request\",0]]";
            if (mode == 10) reply = "[" + aotx_stage_label(0, first) + "," + aotx_stage_label(3, "will read") + "]";
            if (mode == 11) reply = "[" + aotx_stage_item(2, first) + "]";
            if (mode == 12) reply = "[" + aotx_stage_label(0, "Is the door open?") + "," + aotx_stage_label(3, first) + "]";
            if (mode == 13) reply = "[[\"" + first + "\",\"Statement\"]]";
            if (mode == 14) reply = "[[\"" + first + "\",\"\\u0073tatement\"]]";
            if (mode == 15) reply = "[[\"" + first + "\",\"fact\"]]";
            if (mode == 16) reply = "[[\"" + first + "\"]]";
            if (mode == 17) reply = "[[\"" + first + "\",\"state ment\"]]";
            if (mode == 18) reply = aotx_stage_statements(i);
            if (mode == 19) reply = "[[\"" + first + "\",3]]";
            replies.push_back(reply);
        }
        aotx_stage_check(n, 1, replies, mode == 0 || mode == 8);
    }
    aotx_intake_fixture_first_upload(accepted); aotx_intake_fixture_statements<<<1,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned mode = 0; mode < 10; ++mode) {
        std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i) {
            auto first = aotx_stage_first(i), second = aotx_stage_second(i);
            std::string reply = aotx_stage_statements(i);
            if (mode == 1) reply = "[" + aotx_stage_item(3, first) + "]";
            if (mode == 2) reply = "[" + aotx_stage_item(3, second) + "," + aotx_stage_item(3, first) + "]";
            if (mode == 3) reply = "[" + aotx_stage_item(3, "will read") + "," + aotx_stage_item(3, second) + "]";
            if (mode >= 4 && mode < 9) {
                reply.pop_back();
                reply += "," + aotx_stage_item(mode == 4 ? 3 : mode == 8 ? 0 : 1, mode == 4 ? "Is the door open?" :
                    mode == 5 ? "Reply briefly" : mode == 6 ? "will read. Is the door open? Their colleague" : "Person" + std::to_string(i)) + "]";
            }
            if (mode == 9) reply = aotx_stage_extracted(i);
            replies.push_back(reply);
        }
        aotx_stage_check(n, 2, replies, mode == 0 || mode == 7);
    }
}
static void aotx_stage_mutate(aotx_live_records &records, size_t offset, unsigned char delta = 1) {
    for (auto &record : records) {
        auto h = (aotx_record_header *)record.data(); auto b = record.data() + 64;
        if (h->type != 33 || aotx_get(b + 4, 4) != 14) continue;
        auto first = aotx_get(b + 28, 4);
        if (offset >= first && offset < first + h->body_len - 32) { b[32 + offset - first] ^= delta; return; }
    }
    aotx_check(false, "the mutation reaches its recorded byte");
}
static void aotx_stage_replay(unsigned n, unsigned mode) {
    bool skip = mode == 1 || mode == 2;
    aotx_live_records start, records; aotx_bytes expected, choice;
    {
        aotx_intake_device d(n); aotx_fixture empty;
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
        auto bind = d.send(aotx_intake_bind(n), 3); start.insert(start.end(), bind.begin(), bind.end());
        std::vector<std::string> replies, first_outputs;
        for (unsigned i = 0; i < n; ++i) {
            replies.push_back(skip ? "[]" : aotx_stage_statements(i));
            first_outputs.push_back(mode == 2 ? aotx_stage_rejected(i) :
                mode == 3 ? aotx_stage_mixed(i) : skip ? "[]" : aotx_stage_extracted(i));
        }
        aotx_intake_first_outputs = first_outputs;
        records = d.intake(aotx_stage_query(n, mode == 1), replies); expected = aotx_retain_store();
        aotx_check(!d.state().status, "the complete two-stage batch admits atomically");
        if (d.state().status) return;
        choice = aotx_retain_result(records, 14);
        for (unsigned i = 0; i < n; ++i) {
            auto row = choice.data() + 64 + i * AOTX_LIVE_INTAKE_SOURCE_ROW;
            auto meta = row + AOTX_LIVE_AUTO_ROW, first = row + AOTX_LIVE_INTAKE_FIRST;
            aotx_check(aotx_get(first, 4) == 2 && aotx_get(meta, 4) == 2 && aotx_get(meta + 80, 4) == !skip &&
                aotx_get(first + 72, 4) == (skip ? 0u : 2u) && aotx_get(meta + 72, 4) == (skip ? 0u : 2u) &&
                !memcmp(first + 8, meta + 8, 32) && aotx_get(first + 76, 4) == aotx_get(meta + 76, 4) &&
                aotx_get(first + 80, 4) == AOTX_SOURCE_PROFILE && aotx_get(first + 84, 4) == (mode == 1 ? 0u : 4u),
                "both raw outputs retain matching model and role with an explicit empty skip");
            aotx_check(aotx_get(first + 4, 4) == first_outputs[i].size() &&
                !memcmp(first + 128, first_outputs[i].data(), first_outputs[i].size()), "the first raw output is recorded exactly");
        }
    }
    {
        aotx_intake_device d(n); d.process(start, true); d.process(records, true);
        aotx_check(!d.state().fatal && aotx_retain_store() == expected,
            "two-stage decisions restore every canonical state byte without generation");
    }
    for (unsigned mutation = 0; mutation < (mode == 1 ? 0u : mode == 3 ? 1u : 12u); ++mutation) {
        aotx_intake_device d(n); d.process(start, true); auto before = aotx_retain_store(); auto bad = records;
        size_t offsets[] = {AOTX_LIVE_INTAKE_FIRST + 128, AOTX_LIVE_INTAKE_FIRST + 40,
            AOTX_LIVE_INTAKE_FIRST + 8, AOTX_LIVE_INTAKE_FIRST + 72, AOTX_LIVE_INTAKE_FIRST + 76,
            AOTX_LIVE_INTAKE_FIRST + 80, AOTX_LIVE_AUTO_ROW + 80, AOTX_LIVE_AUTO_ROW + 128,
            AOTX_LIVE_INTAKE_FIRST + 84, AOTX_LIVE_INTAKE_FIRST + 88, AOTX_LIVE_INTAKE_FIRST + 120,
            AOTX_LIVE_INTAKE_FIRST};
        aotx_stage_mutate(bad, 64 + (n - 1) * AOTX_LIVE_INTAKE_SOURCE_ROW + offsets[mode == 3 ? 3 : mutation], mode == 3 ? 6 : 1);
        d.process(bad, true);
        aotx_check(d.state().fatal && aotx_retain_store() == before,
            "a changed stage output identity count or execution flag refuses the complete replay batch");
    }
}
__global__ void aotx_stage_release(unsigned n, unsigned mode) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; r->bytes = 0;
    while (aotx_intake_fixture_first[i][r->bytes]) {
        r->reply[r->bytes] = aotx_intake_fixture_first[i][r->bytes]; ++r->bytes;
    }
    r->status = aotx_intake_parse(i); r->state = 3;
    aotx_say.slot[i].wanted = 0; aotx_kv.count[i] = 1;
    aotx_seqs.slot[i].state = AOTX_SEQ_STATE_DONE;
    if (!i) {
        aotx_seqs.live = n; aotx_intake.calls = n;
        if (mode == 1) aotx_live.status = AOTX_COG_DENIED;
        if (mode == 2) aotx_model_wrap[AOTX_MODEL_LANGUAGE].bytes[0] ^= 1;
        if (mode == 3) aotx_model_load.resident[AOTX_MODEL_LANGUAGE].body.digest[0] ^= 1;
    }
}
static void aotx_stage_lease(unsigned n, unsigned mode) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto before = aotx_retain_store();
    d.process(aotx_live_parts(aotx_stage_query(n, mode == 4), 4, d.next_id++), false, false);
    std::vector<std::string> replies;
    for (unsigned i = 0; i < n; ++i) replies.push_back(mode == 4 ? "[]" : mode == 5 ? aotx_stage_rejected(i) : aotx_stage_extracted(i));
    aotx_intake_fixture_first_upload(replies); aotx_stage_release<<<1,64>>>(n, mode);
    aotx_intake_step<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    std::vector<aotx_intake_row> rows(n); std::vector<unsigned> owners(n);
    AOTX_CUDA(cudaMemcpyFromSymbol(rows.data(), aotx_intake, n * sizeof(rows[0]), offsetof(aotx_intake_state, rows)));
    AOTX_CUDA(cudaMemcpyFromSymbol(owners.data(), aotx_intake, n * sizeof(owners[0]), offsetof(aotx_intake_state, row)));
    for (unsigned i = 0; i < n; ++i) {
        aotx_check(mode == 0 ? owners[i] == i + 1 && rows[i].state == 1 && rows[i].phase == 2 &&
            rows[i].first_count == 2 && !rows[i].bytes && !rows[i].ticks && !rows[i].tokens : !owners[i],
            "slot ownership survives first-call release only for a valid second call");
        if (mode == 4 || mode == 5) aotx_check(rows[i].bytes == 2 && !rows[i].second_call && !rows[i].target_count,
            "empty statements release the slot without a second call or correction target");
    }
    unsigned long long calls = 0;
    AOTX_CUDA(cudaMemcpyFromSymbol(&calls, aotx_intake, sizeof(calls), offsetof(aotx_intake_state, calls)));
    aotx_check(calls == n, "the first-call handoff or empty skip starts no additional decoder call");
    aotx_check(aotx_retain_store() == before, "first-call completion and cancellation publish no source prefix");
    aotx_check(mode == 0 ? d.state().phase == AOTX_INTAKE_RUN : d.state().phase == AOTX_INTAKE_DONE,
        "both successful and refused handoffs keep the finite batch state");
}
__global__ void aotx_stage_capacity_open(unsigned n, bool cancel) {
    unsigned i = threadIdx.x; if (i >= n) return;
    if (cancel) { if (!i) aotx_live.status = AOTX_COG_DENIED; return; }
    aotx_say_count[i] = AOTX_SEQ_MAX_TOKENS; aotx_say_gear.piece_count[i] = 1;
    aotx_say_gear.chunk[i * AOTX_SAY_PIECES] = AOTX_SEQ_MAX_TOKENS;
    aotx_intake_open(i);
}
static void aotx_stage_capacity(unsigned n, unsigned mode) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto before = aotx_retain_store();
    d.process(aotx_live_parts(aotx_stage_query(n), 4, d.next_id++), false, false);
    if (mode) {
        std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i) replies.push_back(aotx_stage_extracted(i));
        aotx_intake_fixture_first_upload(replies); aotx_stage_release<<<1,64>>>(n, 0);
        aotx_intake_step<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    }
    aotx_stage_capacity_open<<<1,64>>>(n, mode == 2); aotx_intake_step<<<1,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_check(d.state().phase == AOTX_INTAKE_DONE && d.state().status && aotx_retain_store() == before,
        "either call capacity refusal or between-call cancellation releases the whole batch without publication");
    std::vector<unsigned> owners(n);
    AOTX_CUDA(cudaMemcpyFromSymbol(owners.data(), aotx_intake, n * sizeof(owners[0]), offsetof(aotx_intake_state, row)));
    for (auto owner : owners) aotx_check(!owner, "refused call leaves no source lease");
}
static void aotx_stage_atomic(unsigned n) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto before = aotx_retain_store(); std::vector<std::string> replies;
    for (unsigned i = 0; i < n; ++i) { replies.push_back(aotx_stage_statements(i)); aotx_intake_first_outputs.push_back(aotx_stage_mixed(i)); }
    auto &bad = aotx_intake_first_outputs.back(); bad.pop_back(); bad += "," + aotx_stage_label(0, "will read") + "]";
    auto records = d.intake(aotx_stage_query(n), replies);
    aotx_check(d.state().status && aotx_retain_store() == before,
        "a late rejected-item overlap refuses every source before compaction or publication");
}
static void aotx_stage_legacy(unsigned n) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto query = aotx_stage_query(n);
    for (unsigned i = 0; i < n; ++i) memset(query.data() + 128 + i * AOTX_LIVE_QUERY_ROW + AOTX_RECALL_EXTENSION,
        0, AOTX_RECALL_QUERY - AOTX_RECALL_EXTENSION);
    d.process(aotx_live_parts(query, 4, d.next_id++), false, false);
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    std::vector<std::string> replies(n, "[[0,\"Is the door open?\",0]]"); aotx_stage_check(n, 0, replies, false);
    aotx_stage_check(n, 0, std::vector<std::string>(n, "[[\"Is the door open?\",\"request\"]]"), false);
}
#include "intake_span_capacity.h"
#include "intake_position_replay.h"
int main(int argc, char **argv) {
    const char *cases[] = {"spans", "legacy", "atomic", "replay", "lease", "capacity", "prompts", "spancap", "positions"};
    unsigned only = argc > 1 ? !strcmp(argv[1], "1") ? 1 : !strcmp(argv[1], "64") ? 64 : 0 : 0;
    bool valid = argc <= 3 && (argc == 1 || only);
    if (argc == 3) { bool found = false; for (auto name : cases) found |= !strcmp(argv[2], name); valid &= found; }
    if (!valid) { fprintf(stderr, "usage: aotx_intake_stages_test [1|64] [spans|legacy|atomic|replay|lease|capacity|prompts|spancap|positions]\n"); return 2; }
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    printf("stage sizes row=%zu index=%zu state=%zu live=%zu choice_row=%u\n", sizeof(aotx_intake_row),
        sizeof(aotx_intake_index_row), sizeof(aotx_intake_state), sizeof(aotx_live_state), AOTX_LIVE_INTAKE_SOURCE_ROW);
    for (unsigned n : {1u, 64u}) {
        if (only && only != n) continue;
        for (unsigned which = 0; which < 9; ++which) {
            if (argc == 3 && strcmp(argv[2], cases[which])) continue;
            printf("interpretation stages N=%u case=%s start\n", n, cases[which]); fflush(stdout);
            auto start = std::chrono::steady_clock::now();
            if (!which) aotx_stage_spans(n);
            else if (which == 1) aotx_stage_legacy(n);
            else if (which == 2) aotx_stage_atomic(n);
            else if (which == 3) for (unsigned mode = 0; mode < 4; ++mode) aotx_stage_replay(n, mode);
            else if (which == 4) for (unsigned mode = 0; mode < 6; ++mode) aotx_stage_lease(n, mode);
            else if (which == 5) for (unsigned mode = 0; mode < 3; ++mode) aotx_stage_capacity(n, mode);
            else if (which == 6) aotx_stage_prompts(n);
            else if (which == 7) aotx_stage_span_capacity(n);
            else for (bool optional : {false, true}) aotx_stage_position_replay(n, optional);
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            printf("interpretation stages N=%u case=%s: %u checks, %u failures, %.3f seconds\n",
                n, cases[which], aotx_checks, aotx_failures, seconds); fflush(stdout);
        }
        printf("interpretation stages N=%u: %u checks, %u failures\n", n, aotx_checks, aotx_failures); fflush(stdout);
    }
    return aotx_failures ? 1 : 0;
}
