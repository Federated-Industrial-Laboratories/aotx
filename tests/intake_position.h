/* Purpose: Verify positional quotes and remaining optional token paths.
 * Owns: Repeated spans, exhausted optional kinds and escape boundary controls.
 * Launch shape: N=1 and N=64 through real index, grammar and parser functions.
 * Lifetime: One token fixture without model weights. */
#ifndef AOTX_TEST_INTAKE_POSITION_H
#define AOTX_TEST_INTAKE_POSITION_H
__global__ void aotx_position_clear(unsigned n) {
    unsigned i = blockIdx.x; if (i >= n) return;
    for (unsigned j = threadIdx.x; j < AOTX_INTAKE_CONSUMED; j += blockDim.x) aotx_intake_consumed[i][j] = 0;
}
__global__ void aotx_position_exchange(unsigned n) {
    unsigned i = blockIdx.x * 2; if (i + 1 >= n) return;
    for (unsigned j = threadIdx.x; j < AOTX_INTAKE_CONSUMED; j += blockDim.x) {
        auto value = aotx_intake_consumed[i][j];
        aotx_intake_consumed[i][j] = aotx_intake_consumed[i + 1][j]; aotx_intake_consumed[i + 1][j] = value;
    }
}
__global__ void aotx_position_count(unsigned n, unsigned count) {
    unsigned i = threadIdx.x; if (i < n) aotx_intake.rows[i].source_count = count;
}
__global__ void aotx_position_second(unsigned n) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; r->phase = 2; r->first_count = r->source_count; r->state = 1;
}
__global__ void aotx_position_parse(unsigned n, unsigned *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i; r->bytes = 0;
    while (r->bytes < AOTX_INTAKE_REPLY && aotx_intake_fixture_first[i][r->bytes]) {
        r->reply[r->bytes] = aotx_intake_fixture_first[i][r->bytes]; ++r->bytes;
    }
    out[i] = aotx_intake_parse(i);
}
static aotx_bytes aotx_position_query(unsigned n, bool repeated) {
    auto query = aotx_intake_query(n, 0, 1);
    for (unsigned i = 0; i < n; ++i) {
        auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW; aotx_source_query(q, 8000 + i);
        std::string text = repeated ? "Unit" + std::to_string(i) + "! Unit" + std::to_string(i) + "!" : aotx_source_unit(i);
        memset(q + 4640, 0, 2048); memcpy(q + 4640, text.data(), text.size()); aotx_put(q + 148, text.size(), 4);
    }
    return query;
}
static void aotx_token_position(unsigned n) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    d.process(aotx_live_parts(aotx_position_query(n, true), 4, d.next_id++), false, false);
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    aotx_token_fixture f(n, {"\\u", "\",\"statement\"]", ",[", "\",0]"});
    std::vector<std::string> prefixes(n), complete(n), second(n);
    for (unsigned i = 0; i < n; ++i) {
        auto quote = "Unit" + std::to_string(i) + "!";
        prefixes[i] = "[[\"" + quote;
        auto item = "[\"" + quote + "\",\"statement\"]";
        complete[i] = "[" + item + "," + item + "]";
        item = "[3,\"" + quote + "\",0]";
        second[i] = "[" + item + "," + item;
    }
    f.prefix(prefixes); f.allows("\\", false); f.allows("\\u", false); f.allows("\",\"statement\"]", true);
    for (unsigned i = 0; i < n; ++i) prefixes[i] += "\",\"statement\"]";
    f.prefix(prefixes); f.allows("]", false); f.allows(",[", true);
    aotx_position_count<<<1,64>>>(n, 1); AOTX_CUDA(cudaDeviceSynchronize()); f.allows("]", true);
    auto omitted = prefixes; for (auto &text : omitted) text += ']';
    aotx_intake_fixture_first_upload(omitted); aotx_position_parse<<<1,64>>>(n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(f.out[i], "independent parsing rebuilds coverage after prefix count corruption");
    f.prefix(complete); f.allows(",", false); f.allows("<end>", true);
    aotx_intake_fixture_first_upload(complete); aotx_position_parse<<<1,64>>>(n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(!f.out[i], "repeated whole first quotes parse at their distinct positions");
    aotx_position_second<<<1,64>>>(n); aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    f.prefix(second); f.allows(",", false); f.allows("]", true);
    for (auto &text : second) text += ']';
    aotx_intake_fixture_first_upload(second); aotx_position_parse<<<1,64>>>(n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(!f.out[i], "repeated required second quotes parse without optional uniqueness");
}
static void aotx_token_exhaustion(unsigned n) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    d.process(aotx_live_parts(aotx_position_query(n, false), 4, d.next_id++), false, false);
    aotx_position_second<<<1,64>>>(n); aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    std::vector<std::string> mandatory(n), first(n), last(n), prefixes(n), pieces = {"1", "2", "\\u"};
    std::vector<bool> one(n), two(n);
    for (unsigned i = 0; i < n; ++i) {
        auto quote = aotx_source_unit(i);
        mandatory[i] = "[[3,\"" + quote + "\",0]";
        first[i] = ",[" + std::to_string(1 + i % 2) + ",\"" + quote + "\",0]";
        last[i] = ",[" + std::to_string(2 - i % 2) + ",\"" + quote + "\",0]";
        one[i] = i % 2; two[i] = !one[i];
        for (const auto &tail : {"", " ", "]", ",[", " ,"}) pieces.push_back(first[i] + tail);
    }
    aotx_token_fixture f(n, pieces);
    f.prefix(mandatory);
    for (const auto &tail : {"", " ", "]", ",[", " ,"}) {
        auto candidates = first; for (auto &text : candidates) text += tail;
        f.allows(candidates, std::vector<bool>(n, std::string(tail).find(',') == std::string::npos));
    }
    for (unsigned i = 0; i < n; ++i) prefixes[i] = mandatory[i] + first[i] + ", [";
    f.prefix(prefixes); f.allows(std::vector<std::string>(n, "1"), one); f.allows(std::vector<std::string>(n, "2"), two);
    if (n > 1) {
        aotx_position_exchange<<<n / 2,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
        f.allows(std::vector<std::string>(n, "1"), two); f.allows(std::vector<std::string>(n, "2"), one);
    }
    aotx_position_clear<<<n,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize()); f.allows("1", true); f.allows("2", true);
    for (unsigned i = 0; i < n; ++i) prefixes[i] = mandatory[i] + first[i] + last[i].substr(0, last[i].size() - 4);
    f.prefix(prefixes); f.allows("\\", false); f.allows("\\u", false);
    for (unsigned i = 0; i < n; ++i) prefixes[i] = mandatory[i] + first[i] + last[i];
    f.prefix(prefixes); f.allows(",", false); f.allows("]", true); f.allows(" ", true); f.allows("<end>", false);
    for (auto &text : prefixes) text += ']';
    f.prefix(prefixes); f.allows("<end>", true); f.allows(" ", true);
    aotx_intake_fixture_first_upload(prefixes); aotx_position_parse<<<1,64>>>(n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(!f.out[i], "complete parsing accepts valid bytes independently of token splits");
    for (unsigned i = 0; i < n; ++i) prefixes[i] = mandatory[i] + first[i] + first[i] + "]";
    aotx_intake_fixture_first_upload(prefixes); aotx_position_parse<<<1,64>>>(n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) aotx_check(f.out[i], "independent parsing refuses a consumed optional quote");
}
#endif
