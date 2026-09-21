/* Purpose: Verify optional quotes retain a unique completion inside accepted spans.
 * Owns: Repeated-source cases, an independent substring oracle and sampler checks.
 * Launch shape: Distinct N=1 and N=64 rows through the index and token kernels.
 * Lifetime: One token fixture; no model weights or memory publication. */
#ifndef AOTX_INTAKE_COMPLETION_TEST_H
#define AOTX_INTAKE_COMPLETION_TEST_H
#include <set>

__global__ void aotx_completion_setup(unsigned n, const aotx_intake_span *spans, unsigned count) {
    unsigned i = threadIdx.x; if (i >= n) return;
    auto *r = aotx_intake.rows + i;
    r->phase = 2; r->first_count = count; r->state = 1;
    for (unsigned j = 0; j < count; ++j) r->statements[j] = spans[2 * i + j];
}
__global__ void aotx_completion_read(unsigned n, unsigned width, unsigned char *out) {
    unsigned i = threadIdx.x; if (i >= n) return;
    const auto *s = aotx_intake_index_rows + i;
    for (unsigned start = 0; start < s->bytes; ++start) {
        unsigned node = 0;
        for (unsigned end = start; end < s->bytes; ++end) {
            node = aotx_intake_next(s, node, s->source[end]);
            out[(i * width + start) * width + end - start] =
                end - start + 1 <= aotx_intake_completion[i][node];
        }
    }
}
__global__ void aotx_completion_remove(unsigned n) {
    unsigned i = blockIdx.x; if (i >= n) return;
    for (unsigned j = threadIdx.x; j < aotx_intake_index_rows[i].nodes; j += blockDim.x)
        aotx_intake_completion[i][j] = UINT32_MAX;
}
static void aotx_token_completion(unsigned n, bool extension) {
    aotx_intake_device d(n); aotx_fixture empty;
    d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1); d.send(aotx_intake_bind(n), 3);
    auto query = aotx_intake_query(n, 0, 1);
    std::vector<std::string> sources(n), quotes(n), more(n), prefixes(n), extra = {"is ready.", "is ready", "\",0]]"};
    aotx_intake_span *spans; AOTX_CUDA(cudaMallocManaged(&spans, n * 2 * sizeof(*spans)));
    unsigned width = 0;
    for (unsigned i = 0; i < n; ++i) {
        auto number = std::to_string(i);
        quotes[i] = "The crane" + number + " is ready.";
        more[i] = "The crane" + number + " is ready now.";
        sources[i] = "Please tell whether the crane in room " + number + " is ready. " + quotes[i];
        spans[2 * i] = {(unsigned)(sources[i].size() - quotes[i].size()), (unsigned)quotes[i].size()};
        if (extension) {
            sources[i] += " " + more[i];
            spans[2 * i + 1] = {(unsigned)(sources[i].size() - more[i].size()), (unsigned)more[i].size()};
        }
        prefixes[i] = "[[3,\"" + quotes[i] + "\",0]";
        if (extension) prefixes[i] += ",[3,\"" + more[i] + "\",0]";
        prefixes[i] += ",[2,\"";
        extra.push_back(prefixes[i]); extra.push_back(quotes[i]);
        unsigned char *q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
        aotx_source_query(q, 8000 + i);
        memset(q + 4640, 0, 2048); memcpy(q + 4640, sources[i].data(), sources[i].size());
        aotx_put(q + 148, sources[i].size(), 4); width = std::max(width, (unsigned)sources[i].size());
    }
    d.process(aotx_live_parts(query, 4, d.next_id++), false, false);
    aotx_completion_setup<<<1,64>>>(n, spans, extension ? 2 : 1);
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    unsigned char *actual; AOTX_CUDA(cudaMallocManaged(&actual, n * width * width));
    aotx_completion_read<<<1,64>>>(n, width, actual); AOTX_CUDA(cudaDeviceSynchronize());
    for (unsigned i = 0; i < n; ++i) {
        std::set<std::string> viable;
        for (const auto &span : extension ? std::vector<std::string>{quotes[i], more[i]} : std::vector<std::string>{quotes[i]})
            for (unsigned at = 0; at < span.size(); ++at) for (unsigned bytes = 1; bytes <= span.size() - at; ++bytes) {
                auto text = span.substr(at, bytes); auto first = sources[i].find(text);
                if (sources[i].find(text, first + 1) != std::string::npos) continue;
                for (unsigned prefix = 1; prefix <= bytes; ++prefix) viable.insert(text.substr(0, prefix));
            }
        for (unsigned at = 0; at < sources[i].size(); ++at) for (unsigned bytes = 1; bytes <= sources[i].size() - at; ++bytes)
            aotx_check(actual[(i * width + at) * width + bytes - 1] == viable.count(sources[i].substr(at, bytes)),
                "device completion bounds match independent exact substring enumeration");
    }
    aotx_token_fixture f(n, extra);
    f.prefix(prefixes); f.allows("is ready.", false); f.allows("is ready", extension);
    aotx_completion_remove<<<n,64>>>(n); AOTX_CUDA(cudaDeviceSynchronize());
    f.allows("is ready.", true);
    aotx_intake_index<<<n,64>>>(); AOTX_CUDA(cudaDeviceSynchronize());
    f.allows("is ready.", false);
    f.reset();
    for (unsigned step = 0; step < 4; ++step) {
        std::vector<unsigned> wanted(n);
        for (unsigned i = 0; i < n; ++i) {
            unsigned slot = f.agents[i];
            wanted[i] = f.id(step == 0 ? prefixes[slot] : step == 1 ? quotes[slot] : step == 2 ? "\",0]]" : "<end>");
        }
        f.logits(wanted, false);
        if (step == 1) for (unsigned i = 0; i < n; ++i) f.head[i * f.count + f.id("is ready.")] = 10000.0f;
        f.pick();
        for (unsigned i = 0; i < n; ++i) aotx_check(f.tokens[i] == (int)wanted[i],
            "sampler excludes the repeated suffix despite its greater finite score");
        aotx_token_accept<<<1,64>>>(f.tokens, f.agents, n, f.out); AOTX_CUDA(cudaDeviceSynchronize());
        for (unsigned i = 0; i < n; ++i) aotx_check(f.out[i], "viable quote tokens complete through independent parsing");
    }
    cudaFree(actual); cudaFree(spans);
}
#endif
