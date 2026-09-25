/* Purpose: Check common memory framing through the real device prompt function.
 * Owns: Exact source, current-request, role and capacity assertions.
 * Launch shape: Each CUDA batch contains one or 64 distinct conversation slots.
 * Lifetime: One test process; historical text remains data in every fixture. */
#include "live_context_fixture.h"
#include "cognitive/recall_labels.cuh"

__global__ void aotx_context_render_batch(unsigned char *output, const unsigned char *requests,
    const unsigned *starts, const unsigned *lengths, unsigned *ends, unsigned count, bool append) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    unsigned char *out = output + i * aotx_context_stride + aotx_context_guard;
    unsigned at = aotx_live_memory_rule(i, out, starts[i]);
    at = aotx_live_context(i, out, at);
    ends[2 * i] = at;
    if (append && at <= AOTX_SAY_BYTES) {
        const aotx_wrap *wrap = aotx_wrap_active(aotx_prompt_role(i));
        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_HEAD);
        at = aotx_recall_run(out, at, AOTX_SAY_BYTES, requests + i * 256, lengths[i]);
        at = aotx_wrap_put(out, at, AOTX_SAY_BYTES, wrap, AOTX_WRAP_USER_TAIL);
    }
    ends[2 * i + 1] = at;
}

static void aotx_context_case(unsigned count, unsigned mode, unsigned form, bool compact) {
    aotx_context_device d(count, mode, form, compact);
    for (unsigned boundary = 0; boundary < 7; ++boundary) {
        d.prepare(boundary);
        bool append = boundary == 0 || boundary >= 5;
        aotx_context_render_batch<<<1,64>>>(d.output, d.requests, d.starts, d.lengths, d.ends, count, append);
        AOTX_CUDA(cudaGetLastError()); AOTX_CUDA(cudaDeviceSynchronize());
        std::vector<unsigned char> output(count * aotx_context_stride);
        std::vector<unsigned> ends(2 * count);
        std::vector<aotx_live_binding> after(count);
        AOTX_CUDA(cudaMemcpy(output.data(), d.output, output.size(), cudaMemcpyDeviceToHost));
        AOTX_CUDA(cudaMemcpy(ends.data(), d.ends, ends.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        AOTX_CUDA(cudaMemcpyFromSymbol(after.data(), aotx_live_bindings, count * sizeof(after[0])));
        for (unsigned i = 0; i < count; ++i) {
            const unsigned char *base = output.data() + i * aotx_context_stride;
            const unsigned char *p = base + aotx_context_guard;
            auto frame = d.frame(i); auto request = d.request(i);
            unsigned start = d.offsets[i];
            bool empty = d.sources[i].empty();
            bool fits = empty || (start <= AOTX_SAY_BYTES && frame.size() <= AOTX_SAY_BYTES - start);
            unsigned expected = fits ? start + frame.size() : AOTX_SAY_BYTES + 1;
            aotx_check(ends[2 * i] == expected, "framed length includes all memory and role bytes or reports capacity");
            aotx_check(!memcmp(&d.bindings[i], &after[i], sizeof(after[i])),
                "rendering preserves the complete binding, selected references and historical source bytes");
            aotx_check(std::all_of(base, p, [](unsigned char c) { return c == 0xa5; }) &&
                std::all_of(p + AOTX_SAY_BYTES, base + aotx_context_stride, [](unsigned char c) { return c == 0xa5; }),
                "rendering and refusal preserve both allocation guards");
            unsigned prefix = std::min(start, AOTX_SAY_BYTES);
            aotx_check(std::all_of(p, p + prefix, [](unsigned char c) { return c == 0xa5; }),
                "memory does not change bytes before its supplied prompt offset");
            if (fits && start <= AOTX_SAY_BYTES) {
                std::string wanted = frame + (append ? request : "");
                bool complete = wanted.size() <= AOTX_SAY_BYTES - start;
                aotx_check(ends[2 * i + 1] == (complete ? start + wanted.size() : AOTX_SAY_BYTES + 1),
                    "current request follows the memory boundary or reports its own capacity limit");
                aotx_check(!memcmp(p + start, frame.data(), frame.size()), "the complete memory frame remains exact before current input");
                if (complete) {
                    aotx_check(!memcmp(p + start, wanted.data(), wanted.size()),
                        "model role, historical text, end boundary and current input have exact independent bytes");
                    aotx_check(std::all_of(p + start + wanted.size(), p + AOTX_SAY_BYTES,
                        [](unsigned char c) { return c == 0xa5; }), "successful rendering writes no extra prompt bytes");
                }
                if (!empty) {
                    unsigned source_at = start + d.heads[d.roles[i]].size() +
                        (compact ? aotx_context_rule.size() + std::string("[begin memory records]\n").size() : aotx_context_head.size());
                    aotx_check(!memcmp(p + source_at, d.sources[i].data(), d.sources[i].size()),
                        "historical imperatives, UTF-8 and delimiter-like text remain exact source data");
                }
            } else aotx_check(ends[2 * i + 1] == expected, "refusal or empty context leaves no current-input append");
        }
    }
}

int main(void) {
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned count : {1u, 64u})
        for (unsigned mode = 0; mode < 5; ++mode)
            for (unsigned form = 0; form < 3; ++form)
                for (bool compact : {false, true}) aotx_context_case(count, mode, form, compact);
    printf("live memory context: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
