/* Purpose: Verify token progress and complete constrained sampler responses.
 * Owns: Distinct source rows, byte-token fixtures and exact admission expectations.
 * Launch shape: N=1 and N=64 through the vocabulary and sampler kernels.
 * Lifetime: One test process without model weights. */
#include "intake_fixture.h"
#include "source_fixture.h"
#include "cognitive/intake_token.cuh"

__global__ void aotx_token_setup(unsigned n, float *head, int *tokens, unsigned *agents, unsigned count) {
    unsigned i = threadIdx.x;
    if (i < n) {
        agents[i] = n - i - 1; aotx_intake.row[i] = i + 1;
        aotx_intake.rows[i].status = 0; aotx_intake.rows[i].prefix = {}; aotx_intake.rows[i].bytes = 0;
        aotx_seqs.slot[i].role = AOTX_MODEL_LANGUAGE; aotx_seqs.slot[i].stop = count - 1;
    }
    if (!i) {
        aotx_model[AOTX_MODEL_LANGUAGE].vocab = count; aotx_model_space[AOTX_MODEL_LANGUAGE].head = head;
        auto *r = aotx_model_call + AOTX_MODEL_LANGUAGE; *r = {};
        r->rows = n; r->agent = agents; r->token = tokens;
        aotx_model_wrap[AOTX_MODEL_LANGUAGE].end_count = 1;
        aotx_model_wrap[AOTX_MODEL_LANGUAGE].end_ids[0] = count - 2;
    }
}
__global__ void aotx_token_prefix(const unsigned char *text, const unsigned *lengths, unsigned n, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; r->prefix = {}; r->status = 0; r->bytes = lengths[i];
    for (unsigned j = 0; j < AOTX_INTAKE_CONSUMED; ++j) aotx_intake_consumed[i][j] = 0;
    out[i] = aotx_intake_advance(i, text + i * AOTX_INTAKE_REPLY, lengths[i]);
}
__global__ void aotx_token_probe(const unsigned *tokens, unsigned n, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto before = aotx_intake.rows[i].prefix;
    out[i] = aotx_intake_allows(i, tokens[i]);
    out[n + i] = aotx_cog_equal((const unsigned char *)&before,
        (const unsigned char *)&aotx_intake.rows[i].prefix, sizeof(before));
}
__global__ void aotx_token_accept(const int *tokens, const unsigned *agents, unsigned n, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    unsigned slot = agents[i], token = tokens[i]; auto *r = aotx_intake.rows + slot;
    if (token == aotx_seqs.slot[slot].stop || aotx_wrap_end(AOTX_MODEL_LANGUAGE, token)) {
        out[i] = r->prefix.stage == 11 && !r->status && !aotx_intake_parse(slot); return;
    }
    unsigned bytes = aotx_seq_token_text(token, r->reply + r->bytes, AOTX_INTAKE_REPLY - r->bytes, AOTX_MODEL_LANGUAGE);
    out[i] = bytes && aotx_intake_advance(slot, r->reply + r->bytes, bytes);
    r->bytes += bytes;
}
__global__ void aotx_token_ordinary(unsigned n) {
    unsigned i = threadIdx.x; if (i < n) aotx_intake.row[i] = 0;
}
struct aotx_token_fixture {
    unsigned n, count;
    float *head; int *tokens; unsigned *agents, *out, *lengths, *probe;
    unsigned char *raw, *text; unsigned long long *offset;
    std::vector<std::string> pieces;
    explicit aotx_token_fixture(unsigned rows, const std::vector<std::string> &extra = {}) : n(rows) {
        pieces = {" ", "\t", "\n", "\r", " \n\t\r", "[", "]", ",", "0", "3", "\"",
            " \n[ ", "[ 3, \t\"", "\" \n, ", "0 \t] \r] ", "\\", "t", "n", "u", "00", "e9",
            "\\t", "\\n", "\\u00e9", "\\ud83c", "\\udf72", "has", "Ren", "\xc3\xa9", "\xf0\x9f\x8d\xb2.",
            "absent", "", "0 ", " 0", "  ] ", " \"", "\" ", "\\u0009", "\\u000a", "[]",
            "statement", "request", "sta", "tement", "req", "uest", "Statement", "state ment", "\\u0073tatement"};
        for (unsigned i = 0; i < n; ++i) pieces.push_back("Iris" + std::to_string(i) + ":");
        pieces.insert(pieces.end(), extra.begin(), extra.end());
        pieces.push_back("<end>"); pieces.push_back("<stop>"); count = pieces.size();
        AOTX_CUDA(cudaMallocManaged(&head, n * count * sizeof(float)));
        AOTX_CUDA(cudaMallocManaged(&tokens, n * sizeof(int))); AOTX_CUDA(cudaMallocManaged(&agents, n * sizeof(unsigned)));
        AOTX_CUDA(cudaMallocManaged(&out, n * 2 * sizeof(unsigned))); AOTX_CUDA(cudaMallocManaged(&probe, n * sizeof(unsigned)));
        AOTX_CUDA(cudaMallocManaged(&lengths, n * sizeof(unsigned))); AOTX_CUDA(cudaMallocManaged(&text, n * AOTX_INTAKE_REPLY));
        AOTX_CUDA(cudaMallocManaged(&offset, (count + 1) * sizeof(unsigned long long)));
        std::string encoded;
        for (unsigned j = 0; j < count; ++j) {
            offset[j] = encoded.size();
            for (unsigned char byte : pieces[j]) {
                unsigned point = byte;
                if (byte <= 32) point = 256 + byte;
                else if (byte >= 127 && byte <= 160) point = 289 + byte - 127;
                else if (byte == 173) point = 323;
                if (point < 128) encoded += char(point);
                else { encoded += char(192 | (point >> 6)); encoded += char(128 | (point & 63)); }
            }
        }
        offset[count] = encoded.size(); AOTX_CUDA(cudaMallocManaged(&raw, encoded.size()));
        memcpy(raw, encoded.data(), encoded.size());
        aotx_text_vocab vocab = {}; vocab.tokens = count; vocab.token_at = offset; vocab.token_bytes = raw;
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_text_vocab_table, &vocab, sizeof(vocab)));
        reset();
    }
    ~aotx_token_fixture() {
        cudaFree(head); cudaFree(tokens); cudaFree(agents); cudaFree(out); cudaFree(lengths);
        cudaFree(probe); cudaFree(raw); cudaFree(text); cudaFree(offset);
    }
    unsigned id(const std::string &value) const {
        for (unsigned j = 0; j < count; ++j) if (pieces[j] == value) return j;
        fprintf(stderr, "missing token\n"); exit(1);
    }
    void reset() {
        aotx_token_setup<<<1,64>>>(n, head, tokens, agents, count); AOTX_CUDA(cudaDeviceSynchronize());
    }
    void prefix(const std::vector<std::string> &values) {
        for (unsigned i = 0; i < n; ++i) { lengths[i] = values[i].size(); memcpy(text + i * AOTX_INTAKE_REPLY, values[i].data(), lengths[i]); }
        aotx_token_prefix<<<1,64>>>(text, lengths, n, out); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) aotx_check(out[i], "fixture prefix is valid for its distinct source");
    }
    void allows(const std::string &piece, bool expected) {
        allows(std::vector<std::string>(n, piece), std::vector<bool>(n, expected));
    }
    void allows(const std::vector<std::string> &values, const std::vector<bool> &expected) {
        for (unsigned i = 0; i < n; ++i) probe[i] = id(values[i]);
        aotx_token_probe<<<1,64>>>(probe, n, out); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) {
            aotx_check(out[i] == expected[i], "decoded token has the declared admission result");
            aotx_check(out[n + i], "candidate admission leaves the persistent prefix unchanged");
        }
    }
    void logits(const std::vector<unsigned> &wanted, bool outside = true) {
        for (unsigned i = 0; i < n; ++i) {
            for (unsigned j = 0; j < count; ++j) head[i * count + j] = -1000.0f - j;
            for (unsigned j = 0; j < 5; ++j) head[i * count + j] = (outside ? 1000.0f : -100.0f) + (j + i) % 5;
            head[i * count + id("absent")] = 2000.0f + i;
            head[i * count + wanted[i]] = wanted[i] < 5 ? 1100.0f + i : 100.0f + i;
        }
    }
    void pick() { aotx_model_pick<<<n,AOTX_MODEL_ROW_THREADS>>>(AOTX_MODEL_LANGUAGE); AOTX_CUDA(cudaDeviceSynchronize()); }
};
#include "intake_completion.h"
#include "intake_position.h"
static std::string aotx_token_source(unsigned i) {
    return "Iris" + std::to_string(i) + ": has\t\nRen\xc3\xa9 \xf0\x9f\x8d\xb2.";
}
static std::vector<std::string> aotx_token_prefixes(unsigned n, const std::string &tail, bool quoted = true) {
    std::vector<std::string> rows;
    for (unsigned i = 0; i < n; ++i) rows.push_back(quoted ? "[[3,\"Iris" + std::to_string(i) + ":" + tail : tail);
    return rows;
}
static void aotx_token_direct(aotx_token_fixture &f) {
    for (const std::string &prefix : {"", "[", "[[", "[[3", "[[3,", "[[3,\"Iris@:\"", "[[3,\"Iris@:\",",
        "[[3,\"Iris@:\",0", "[[3,\"Iris@:\",0 ", "[[3,\"Iris@:\",0]", "[[3,\"Iris@:\",0],", "[]"}) {
        std::vector<std::string> rows;
        for (unsigned i = 0; i < f.n; ++i) { auto p = prefix; auto at = p.find('@'); if (at != std::string::npos) p.replace(at, 1, std::to_string(i)); rows.push_back(p); }
        f.prefix(rows);
        for (unsigned j = 0; j < 5; ++j) f.allows(f.pieces[j], false);
        f.allows("", false); f.allows("<stop>", prefix == "[]"); f.allows("<end>", prefix == "[]");
    }
    f.prefix(aotx_token_prefixes(f.n, "", false)); f.allows(" \n[ ", true);
    f.prefix(aotx_token_prefixes(f.n, "[[3,", false)); f.allows(" \"", true);
    f.prefix(aotx_token_prefixes(f.n, "")); f.allows(" ", true); f.allows("\" ", true); f.allows("\t", false);
    f.prefix(aotx_token_prefixes(f.n, " has")); f.allows("\\t", true); f.allows("\\u0009", true);
    f.prefix(aotx_token_prefixes(f.n, " has\\t")); f.allows("\\n", true); f.allows("\\u000a", true);
    f.prefix(aotx_token_prefixes(f.n, " has\\t\\nRen")); f.allows("\xc3\xa9", true); f.allows("\\u00e9", true);
    f.prefix(aotx_token_prefixes(f.n, " has\\t\\nRen\\")); f.allows("u", true);
    f.prefix(aotx_token_prefixes(f.n, " has\\t\\nRen\\u")); f.allows("00", true);
    f.prefix(aotx_token_prefixes(f.n, " has\\t\\nRen\\u00")); f.allows("e9", true);
    f.prefix(aotx_token_prefixes(f.n, " has\\t\\nRen\\u00e9 ")); f.allows("\\ud83c", true);
    f.prefix(aotx_token_prefixes(f.n, " has\\t\\nRen\\u00e9 \\ud83c")); f.allows("\\udf72", true);
    f.prefix(aotx_token_prefixes(f.n, "\",0")); f.allows("  ] ", true); f.allows("0", false);
    f.prefix(aotx_token_prefixes(f.n, "\",")); f.allows("0 ", true); f.allows(" 0", true); f.allows("0 \t] \r] ", true);
}
static void aotx_token_complete(aotx_token_fixture &f) {
    f.reset();
    std::vector<std::string> received(f.n);
    for (unsigned step = 0; step < 19; ++step) {
        std::vector<unsigned> wanted(f.n);
        for (unsigned i = 0; i < f.n; ++i) {
            unsigned slot = f.agents[i];
            std::vector<std::string> path = {" \n[ ", "[", "3", ",", " \"", "Iris" + std::to_string(slot) + ":", " ", "has",
                "\\t", "\\n", "Ren", slot % 2 ? "\\u00e9" : "\xc3\xa9", " ", "\xf0\x9f\x8d\xb2.",
                "\" \n, ", " 0", "  ] ", "]", slot % 2 ? "<end>" : "<stop>"};
            wanted[i] = f.id(path[step]); if (step < 18) received[slot] += path[step];
        }
        f.logits(wanted, step != 5); f.pick();
        for (unsigned i = 0; i < f.n; ++i) {
            if (f.tokens[i] != (int)wanted[i]) fprintf(stderr, "row %u step %u: selected %d expected %u\n", i, step, f.tokens[i], wanted[i]);
            aotx_check(f.tokens[i] == (int)wanted[i], "sampler selects progress above all valid finite alternatives");
        }
        aotx_token_accept<<<1,64>>>(f.tokens, f.agents, f.n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < f.n; ++i) aotx_check(f.out[i], "selected tokens advance through independent complete parsing");
    }
    for (unsigned i = 0; i < f.n; ++i) {
        aotx_intake_row row; AOTX_CUDA(cudaMemcpyFromSymbol(&row, aotx_intake, sizeof(row),
            offsetof(aotx_intake_state, rows) + i * sizeof(row)));
        aotx_check(!row.status && row.prefix.stage == 11 && row.count == 1 && row.bytes == received[i].size() &&
            !memcmp(row.reply, received[i].data(), row.bytes), "each completed response preserves its exact selected bytes");
        auto source = aotx_token_source(i);
        aotx_check(row.items[0].start == 0 && row.items[0].length == source.size(), "complete parsing retains the full distinct source span");
    }
    f.prefix(aotx_token_prefixes(f.n, "[]", false));
    for (unsigned i = 0; i < f.n; ++i) for (unsigned j = 0; j < f.count; ++j)
        f.head[i * f.count + j] = j < 5 ? 1000.0f + i : -INFINITY;
    f.pick();
    for (unsigned i = 0; i < f.n; ++i) {
        unsigned status = 0; AOTX_CUDA(cudaMemcpyFromSymbol(&status, aotx_intake, sizeof(status),
            offsetof(aotx_intake_state, rows) + i * sizeof(aotx_intake_row) + offsetof(aotx_intake_row, status)));
        aotx_check(f.tokens[i] == (int)f.count - 1 && status == AOTX_COG_CAPACITY, "all masked output refuses each complete row");
    }
    f.reset(); aotx_token_ordinary<<<1,64>>>(f.n); AOTX_CUDA(cudaDeviceSynchronize());
    std::vector<unsigned> wanted(f.n, f.id("[")); f.logits(wanted); f.pick();
    for (unsigned i = 0; i < f.n; ++i) aotx_check(f.tokens[i] == (int)f.id("absent"), "ordinary sampling retains its unmasked maximum");
}
static void aotx_token_empty(aotx_token_fixture &f, bool modern) {
    for (bool fused : {false, true}) for (bool end : {false, true}) {
        f.reset();
        std::vector<std::string> path = fused ? std::vector<std::string>{"[]"} : std::vector<std::string>{"[", "]"};
        path.push_back(end ? "<end>" : "<stop>");
        for (const auto &piece : path) {
            f.allows(piece, true);
            std::vector<unsigned> wanted(f.n, f.id(piece)); f.logits(wanted, !modern); f.pick();
            for (unsigned i = 0; i < f.n; ++i) aotx_check(f.tokens[i] == (int)wanted[i],
                "empty output tokens pass live sampling above invalid finite maxima");
            aotx_token_accept<<<1,64>>>(f.tokens, f.agents, f.n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < f.n; ++i) aotx_check(f.out[i],
                "each empty output token advances and the end token completes independent parsing");
        }
        for (unsigned i = 0; i < f.n; ++i) {
            aotx_intake_row row; AOTX_CUDA(cudaMemcpyFromSymbol(&row, aotx_intake, sizeof(row),
                offsetof(aotx_intake_state, rows) + i * sizeof(row)));
            aotx_check(!row.status && row.prefix.stage == 11 && !row.prefix.items && !row.count &&
                row.bytes == 2 && !memcmp(row.reply, "[]", 2),
                "split and combined empty output tokens complete with exactly zero parsed items");
        }
    }
}
static void aotx_token_pairs(aotx_token_fixture &f) {
    for (bool statement : {false, true}) for (bool split : {false, true}) {
        f.reset();
        std::vector<std::vector<std::string>> paths(f.n);
        std::vector<std::string> received(f.n);
        for (unsigned slot = 0; slot < f.n; ++slot) {
            paths[slot] = {"[", "[", "\"", "Iris" + std::to_string(slot) + ":", " ", "has", "\\t",
                "Ren", "\\u00e9", " ", "\xf0\x9f\x8d\xb2.", "\"", ",", "\""};
            paths[slot].push_back(statement ? split ? "sta" : "statement" : split ? "req" : "request");
            if (split) paths[slot].push_back(statement ? "tement" : "uest");
            paths[slot].insert(paths[slot].end(), {"\"", "]", "]", slot % 2 ? "<end>" : "<stop>"});
        }
        for (unsigned step = 0; step < paths[0].size(); ++step) {
            if (step == 14) {
                for (const auto &piece : {" ", "Statement", "state ment", "\\u0073tatement"}) f.allows(piece, false);
                f.allows("statement", true); f.allows("request", true); f.allows("<end>", false);
            }
            std::vector<unsigned> wanted(f.n);
            for (unsigned i = 0; i < f.n; ++i) {
                unsigned slot = f.agents[i]; wanted[i] = f.id(paths[slot][step]);
                if (step + 1 < paths[slot].size()) received[slot] += paths[slot][step];
            }
            f.logits(wanted, false); f.pick();
            for (unsigned i = 0; i < f.n; ++i) aotx_check(f.tokens[i] == (int)wanted[i],
                "split and complete labels pass live token selection");
            aotx_token_accept<<<1,64>>>(f.tokens, f.agents, f.n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
            for (unsigned i = 0; i < f.n; ++i) aotx_check(f.out[i], "pair tokens complete through independent parsing");
        }
        for (unsigned i = 0; i < f.n; ++i) {
            aotx_intake_row row; AOTX_CUDA(cudaMemcpyFromSymbol(&row, aotx_intake, sizeof(row),
                offsetof(aotx_intake_state, rows) + i * sizeof(row)));
            aotx_check(!row.status && row.prefix.stage == 11 && row.count == 1 && row.items[0].kind == (statement ? 3u : 0u) &&
                !row.items[0].target && !row.items[0].start && row.items[0].length == aotx_token_source(i).size() - 1 &&
                row.bytes == received[i].size() && !memcmp(row.reply, received[i].data(), row.bytes),
                "both exact labels retain their source quote and internal kind");
        }
    }
}
static void aotx_token_run(unsigned n, bool modern, bool empty_only) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto p = aotx_intake_query(n, 0, 1);
    for (unsigned i = 0; i < n; ++i) {
        auto q = p.data() + 128 + i * AOTX_LIVE_QUERY_ROW; auto source = aotx_token_source(i);
        if (modern) { aotx_source_query(q, 8000 + i); source.erase(source.find('\n'), 1); }
        memset(q + 4640, 0, 2048); memcpy(q + 4640, source.data(), source.size()); aotx_put(q + 148, source.size(), 4);
    }
    d.process(aotx_live_parts(p, 4, d.next_id++), false, false);
    aotx_check(d.state().phase == AOTX_INTAKE_RUN, "distinct source rows reach internal sampling");
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_token_fixture f(n);
    if (!empty_only) { aotx_token_direct(f); aotx_token_complete(f); }
    if (!modern) aotx_token_empty(f, false);
    if (modern) {
        aotx_token_pairs(f);
        f.prefix(std::vector<std::string>(n, "[        ")); f.allows(" ", false); f.allows("]", false);
        f.prefix(std::vector<std::string>(n, "[    ")); f.allows(" \n\t\r", true);
        f.prefix(std::vector<std::string>(n, "[     ")); f.allows(" \n\t\r", false);
    }
}
static void aotx_token_whitespace(unsigned n) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto query = aotx_intake_query(n, 0, 1);
    for (unsigned i = 0; i < n; ++i) {
        auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW; aotx_source_query(q, 8000 + i);
        memset(q + 4640, 0, 2048); memcpy(q + 4640, " \t\n ", 4); aotx_put(q + 148, 4, 4);
    }
    d.process(aotx_live_parts(query, 4, d.next_id++), false, false);
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_token_fixture f(n); aotx_token_empty(f, true);
}
static void aotx_token_processor(unsigned n) {
    const unsigned char prior[32] = {
        0xfd, 0x01, 0xa6, 0x1c, 0x7c, 0x4c, 0x34, 0xa6, 0xbb, 0x64, 0x79, 0x64, 0x32, 0xf5, 0xd5, 0x5b,
        0xf3, 0x0e, 0x74, 0xc6, 0xc0, 0xf4, 0xab, 0xce, 0xa0, 0xae, 0x35, 0x71, 0x60, 0x4b, 0x61, 0xda
    };
    const unsigned char current[32] = {
        0x0d, 0xd1, 0x2d, 0x23, 0x29, 0xab, 0x9f, 0xc0, 0xee, 0x60, 0x59, 0x10, 0x54, 0x12, 0x59, 0x26,
        0x74, 0x9c, 0xbb, 0x58, 0xad, 0xb6, 0x4a, 0x26, 0x9b, 0x3b, 0x64, 0x77, 0xcb, 0xa6, 0x51, 0x49
    };
    aotx_live_records start; aotx_bytes choice, expected;
    {
        aotx_intake_device d(n); aotx_fixture empty;
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), AOTX_LIVE_LOAD);
        auto bind = d.send(aotx_intake_bind(n), AOTX_LIVE_BIND); start.insert(start.end(), bind.begin(), bind.end());
        choice = aotx_retain_result(d.intake(aotx_intake_query(n, 0, 1), aotx_intake_initial(n)), AOTX_INTAKE_CHOICE);
        expected = aotx_retain_store();
        for (unsigned i = 0; i < n; ++i) aotx_check(!memcmp(choice.data() + 64 + i * AOTX_LIVE_INTAKE_ROW +
            AOTX_LIVE_AUTO_ROW + 40, current, 32), "new decisions record the current generation processor");
    }
    for (unsigned mode = 0; mode < 3; ++mode) {
        auto recorded = choice, store = expected;
        auto *s = (aotx_cognitive_store *)store.data();
        unsigned changed = 0;
        for (unsigned i = 0; i < n; ++i) {
            bool previous = mode == 1 || (mode == 2 && i % 2 == 0);
            if (!previous) continue;
            memcpy(recorded.data() + 64 + i * AOTX_LIVE_INTAKE_ROW + AOTX_LIVE_AUTO_ROW + 40, prior, 32);
            unsigned char *tail = recorded.data() + 64 + n * AOTX_LIVE_INTAKE_ROW;
            unsigned objects = aotx_get(tail + 20, 4);
            for (unsigned j = 0; j < 3; ++j) {
                unsigned at = 3 * n + i * 3 + j;
                auto *r = tail + AOTX_COG_HEADER + at * AOTX_COG_OBJECT;
                auto *p = tail + AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + aotx_get(r + AOTX_CO_OFFSET);
                aotx_check(!memcmp(p + 56, current, 32), "recorded inferred payload has the current processor before replacement");
                memcpy(p + 56, prior, 32);
                auto *saved = s->payload + aotx_get(s->objects[at] + AOTX_CO_OFFSET);
                memcpy(saved + 56, prior, 32); ++changed;
            }
        }
        aotx_check(changed == (mode == 1 ? 3 * n : mode == 2 ? 3 * ((n + 1) / 2) : 0),
            "recovery fixture changes exactly the selected processor payloads");
        {
            aotx_intake_device d(n); d.process(start, true);
            d.process(aotx_live_parts(aotx_intake_query(n, 0, 1), AOTX_LIVE_QUERY, 3), true);
            d.process(aotx_live_parts(recorded, AOTX_INTAKE_CHOICE, 3), true);
            aotx_check(!d.state().fatal && d.state().replays == n, "current previous and mixed processor decisions recover");
            aotx_check(aotx_retain_store() == store, "recovery preserves every store byte and processor digest");
            unsigned long long calls = 1;
            AOTX_CUDA(cudaMemcpyFromSymbol(&calls, aotx_intake, sizeof(calls), offsetof(aotx_intake_state, calls)));
            aotx_check(!calls, "processor recovery starts no generation");
        }
        for (unsigned bad = 0; bad < 2; ++bad) {
            aotx_intake_device d(n); d.process(start, true); auto before = aotx_retain_store();
            auto damaged = recorded;
            if (bad) {
                damaged[64 + AOTX_LIVE_AUTO_ROW + 40] ^= 1;
                d.process(aotx_live_parts(aotx_intake_query(n, 0, 1), AOTX_LIVE_QUERY, 3), true);
            }
            d.process(aotx_live_parts(damaged, AOTX_INTAKE_CHOICE, 3), bad != 0);
            aotx_check(d.state().refused && (!bad || d.state().fatal) && aotx_retain_store() == before,
                "unknown processors and submitted decisions cannot change memory");
        }
    }
}
int main(int argc, char **argv) {
    unsigned only = argc > 1 ? !strcmp(argv[1], "1") ? 1 : !strcmp(argv[1], "64") ? 64 : 0 : 0;
    bool empty_only = argc == 3 && !strcmp(argv[2], "empty");
    bool positions_only = argc == 3 && !strcmp(argv[2], "positions");
    if (argc > 3 || (argc > 1 && !only) || (argc == 3 && !empty_only && !positions_only)) {
        fprintf(stderr, "usage: aotx_intake_token_test [1|64] [empty|positions]\n"); return 2;
    }
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        if (only && only != n) continue;
        unsigned checks = aotx_checks, failures = aotx_failures;
        if (positions_only) {
            aotx_token_position(n); aotx_token_exhaustion(n);
        } else {
            aotx_token_run(n, false, empty_only); aotx_token_run(n, true, true); aotx_token_whitespace(n);
            if (!empty_only) for (bool extension : {false, true}) aotx_token_completion(n, extension);
            if (!empty_only) { aotx_token_position(n); aotx_token_exhaustion(n); aotx_token_processor(n); }
        }
        printf("interpretation tokens N=%u: %u checks, %u failures\n", n, aotx_checks - checks, aotx_failures - failures); fflush(stdout);
    }
    return aotx_failures ? 1 : 0;
}
