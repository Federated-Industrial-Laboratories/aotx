/* Purpose: Check native admission and independent policy rows at both batch sizes.
 * Owns: Distinct observations, state canaries and independent expected decisions.
 * Launch shape: Real compiled CUDA and PTX nodes over N=1 and N=64 rows.
 * Lifetime: One process with temporary bundles and finite graphs. */
#include "policy_fixture.h"

static void aotx_policy_batch(unsigned n, unsigned mode, const char *image, unsigned format, unsigned stride) {
    aotx_policy_asset asset(mode, stride, "aotx_creator_maintenance", image, format); asset.open();
    std::vector<aotx_policy_input> input(n);
    std::vector<aotx_policy_output> output(n);
    aotx_bytes state((size_t)n * stride), next(state.size());
    aotx_policy_input *device_input; aotx_policy_output *device_output;
    unsigned char *device_state, *device_next;
    AOTX_CUDA(cudaMalloc(&device_input, n * sizeof(input[0])));
    AOTX_CUDA(cudaMalloc(&device_output, n * sizeof(output[0])));
    AOTX_CUDA(cudaMalloc(&device_state, state.size())); AOTX_CUDA(cudaMalloc(&device_next, next.size()));
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    AOTX_CUDA(cudaStreamCreate(&stream)); AOTX_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    aotx_check(aotx_policy_rows_capture(stream, device_input, device_state, device_output, device_next, n, stride) == 1,
        "exact selected module adds one real batch node");
    AOTX_CUDA(cudaStreamEndCapture(stream, &graph)); AOTX_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    cudaEvent_t begin, end; AOTX_CUDA(cudaEventCreate(&begin)); AOTX_CUDA(cudaEventCreate(&end));
    float maximum_ms = 0;
    for (unsigned scenario = 0; scenario < 11; ++scenario) {
        for (unsigned i = 0; i < n; ++i) {
            unsigned kind = (scenario + i) % 11;
            auto &r = input[i]; r = {};
            r.valid = kind != 4; r.enabled = kind != 6; r.source = 10000 + 100 * i;
            r.previous_source = r.source - 1; r.objects = kind == 1 || kind == 9 ? 10 : 90;
            r.object_capacity = 100; r.bytes = kind == 9 ? 900 : 1; r.byte_capacity = 1000;
            r.foreground = kind == 2; r.paused = kind == 5;
            r.pressure = kind == 8 ? 0 : kind == 10 ? 40 : 95;
            r.rule_pressure = kind == 8 || kind == 10 ? 0 : 40;
            r.minimum_move = kind == 7 ? 20000 : 1; r.backoff = 8;
            for (unsigned j = 0; j < stride; ++j) state[(size_t)i * stride + j] = (unsigned char)(i * 13 + j * 7);
            aotx_put(state.data() + (size_t)i * stride, 200 + i);
            aotx_put(state.data() + (size_t)i * stride + 8, kind == 3 ? r.source - 2 : 100 + i);
            output[i] = {}; output[i].action = 1234; output[i].reason = 99000 + i;
        }
        auto expected_output = output;
        std::fill(next.begin(), next.end(), 0xa9);
        AOTX_CUDA(cudaMemcpy(device_input, input.data(), n * sizeof(input[0]), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(device_output, output.data(), n * sizeof(output[0]), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(device_state, state.data(), state.size(), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(device_next, next.data(), next.size(), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaEventRecord(begin, stream)); AOTX_CUDA(cudaGraphLaunch(exec, stream));
        AOTX_CUDA(cudaEventRecord(end, stream)); AOTX_CUDA(cudaEventSynchronize(end));
        float elapsed = 0; AOTX_CUDA(cudaEventElapsedTime(&elapsed, begin, end));
        maximum_ms = std::max(maximum_ms, elapsed);
        AOTX_CUDA(cudaMemcpy(output.data(), device_output, n * sizeof(output[0]), cudaMemcpyDeviceToHost));
        AOTX_CUDA(cudaMemcpy(next.data(), device_next, next.size(), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < n; ++i) {
            unsigned kind = (scenario + i) % 11;
            bool maintain = kind == 0 || kind == 9 || kind == 10;
            if (kind == 4) {
                aotx_check(!memcmp(&output[i], &expected_output[i], sizeof(output[i])), "inactive rows leave output untouched");
                aotx_check(std::all_of(next.begin() + (size_t)i * stride, next.begin() + (size_t)(i + 1) * stride,
                    [](unsigned char c) { return c == 0xa9; }), "inactive rows leave private state untouched");
                continue;
            }
            aotx_check(output[i].action == (maintain ? AOTX_POLICY_MAINTAIN : AOTX_POLICY_QUIET),
                "pressure, foreground and state backoff produce the specified independent decision");
            aotx_check(!output[i].status && output[i].reason == (maintain ? AOTX_POLICY_REASON_PRESSURE : 0),
                "each proposal has the expected reason and success state");
            aotx_check(aotx_get(next.data() + (size_t)i * stride) == 201 + i,
                "each distinct private counter advances once");
            uint64_t expected = maintain ? input[i].source : kind == 3 ? input[i].source - 2 : 100 + i;
            aotx_check(aotx_get(next.data() + (size_t)i * stride + 8) == expected,
                "maintenance state advances only for its own admitted proposal");
            aotx_check(!memcmp(next.data() + (size_t)i * stride + 16, state.data() + (size_t)i * stride + 16, stride - 16),
                "all remaining declared state bytes keep their distinct values");
        }
    }
    printf("policy batch n=%u mode=%u format=%u state=%u maximum_ms=%.6f\n", n, mode, format, stride, maximum_ms);
    cudaEventDestroy(end); cudaEventDestroy(begin);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    cudaFree(device_input); cudaFree(device_output); cudaFree(device_state); cudaFree(device_next); aotx_policy_close();
}
static void aotx_policy_refusals(void) {
    aotx_policy_asset good(AOTX_POLICY_NATIVE);
    aotx_check(aotx_policy_open(good.path.c_str(), nullptr) != 0, "native code requires a local exact-digest grant");
    aotx_check(aotx_policy_open(good.path.c_str(), std::string(64, '0').c_str()) != 0, "a different policy digest grants no trust");
    aotx_policy_asset wrong(AOTX_POLICY_NATIVE, 16, "aotx_policy_wrong", AOTX_POLICY_TEST_CASES);
    aotx_check(aotx_policy_open(wrong.path.c_str(), wrong.trust.c_str()) != 0, "wrong native parameter ABI is refused");
    aotx_policy_asset missing(AOTX_POLICY_NATIVE, 16, "aotx_absent");
    aotx_check(aotx_policy_open(missing.path.c_str(), missing.trust.c_str()) != 0, "missing native entry is refused");
    aotx_policy_asset resources(AOTX_POLICY_NATIVE, 16, "aotx_creator_maintenance", AOTX_POLICY_TEST_PTX, 1, 1);
    aotx_check(aotx_policy_open(resources.path.c_str(), resources.trust.c_str()) != 0, "declared register bound has a real consumer");
    aotx_policy_asset target(AOTX_POLICY_NATIVE, 16, "aotx_creator_maintenance", AOTX_POLICY_TEST_PTX, 1, 255, AOTX_ARCH + 1);
    aotx_check(aotx_policy_open(target.path.c_str(), target.trust.c_str()) != 0, "wrong native target is refused before activation");
    good.open(); aotx_policy_close();
}
int main() {
    AOTX_CUDA(cudaFree(nullptr)); aotx_policy_refusals();
    for (unsigned n : {1u, 64u}) {
        aotx_policy_batch(n, AOTX_POLICY_SUPPLIED, AOTX_POLICY_TEST_PTX, 1, 16);
        aotx_policy_batch(n, AOTX_POLICY_RULES, AOTX_POLICY_TEST_PTX, 1, 16);
        aotx_policy_batch(n, AOTX_POLICY_NATIVE, AOTX_POLICY_TEST_PTX, 1, 16);
        aotx_policy_batch(n, AOTX_POLICY_NATIVE, AOTX_POLICY_TEST_CUBIN, 2, AOTX_POLICY_STATE_BYTES);
    }
    printf("policy batch: %u checks, %u failures\n", aotx_checks, aotx_failures);
    return aotx_failures ? 1 : 0;
}
