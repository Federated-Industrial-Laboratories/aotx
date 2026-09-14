/* Purpose: Check versioned appraisal proposals over distinct data and native rows.
 * Owns: Exact expected output and portable state bytes for each row.
 * Launch shape: Finite selected policy nodes at N=1 and N=64.
 * Lifetime: One test-owned bundle and graph per batch. */
#ifndef AOTX_POLICY_APPRAISAL_BATCH_H
#define AOTX_POLICY_APPRAISAL_BATCH_H
#include "policy_fixture.h"
static void aotx_policy_appraisal_batch(unsigned n, unsigned abi, unsigned mode) {
    aotx_policy_asset asset(mode, 16, "aotx_policy_appraisal_native", AOTX_POLICY_TEST_CASES,
        1, 255, AOTX_ARCH, abi); asset.open();
    std::vector<aotx_policy_input> input(n);
    std::vector<aotx_policy_output> output(n);
    aotx_bytes before(16 * n), after(16 * n), expected;
    aotx_policy_input *di; aotx_policy_output *dout; unsigned char *db, *da;
    AOTX_CUDA(cudaMalloc(&di, n * sizeof(input[0]))); AOTX_CUDA(cudaMalloc(&dout, n * sizeof(output[0])));
    AOTX_CUDA(cudaMalloc(&db, before.size())); AOTX_CUDA(cudaMalloc(&da, after.size()));
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t exec;
    AOTX_CUDA(cudaStreamCreate(&stream)); AOTX_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    aotx_check(aotx_policy_rows_capture(stream, di, db, dout, da, n, 16) == 1, "the selected version captures a real batch node");
    AOTX_CUDA(cudaStreamEndCapture(stream, &graph)); AOTX_CUDA(cudaGraphInstantiate(&exec, graph, 0));
    for (unsigned scenario = 0; scenario < 8; ++scenario) {
        for (unsigned i = 0; i < n; ++i) {
            unsigned kind = (i + scenario) % 8;
            auto &in = input[i]; in = {};
            in.valid = kind != 4; in.enabled = kind != 6; in.source = 8000 + i * 91;
            in.objects = kind == 5 ? 90 : 10; in.object_capacity = 100;
            in.bytes = 10 + i; in.byte_capacity = 1000; in.pressure = 40; in.minimum_move = 1; in.backoff = 1;
            in.paused = kind == 3; in.foreground = kind == 2;
            if (abi == 2) {
                uint64_t revision = 0x100000000ull + i * 17 + scenario * 101 + 3;
                in.reserved0 = 2; in.reserved1[0] = kind == 1 ? 0 : kind == 7 ? 1 : 2 + i;
                in.reserved1[1] = (uint32_t)revision; in.reserved1[2] = (uint32_t)(revision >> 32);
            }
            aotx_put(before.data() + 16 * i, 200 + i); aotx_put(before.data() + 16 * i + 8, 100 + i);
            output[i] = {}; output[i].action = 999; output[i].reason = 77000 + i;
        }
        auto original = output; std::fill(after.begin(), after.end(), 0xa9); expected = after;
        AOTX_CUDA(cudaMemcpy(di, input.data(), n * sizeof(input[0]), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(dout, output.data(), n * sizeof(output[0]), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(db, before.data(), before.size(), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaMemcpy(da, after.data(), after.size(), cudaMemcpyHostToDevice));
        AOTX_CUDA(cudaGraphLaunch(exec, stream)); AOTX_CUDA(cudaStreamSynchronize(stream));
        AOTX_CUDA(cudaMemcpy(output.data(), dout, n * sizeof(output[0]), cudaMemcpyDeviceToHost));
        AOTX_CUDA(cudaMemcpy(after.data(), da, after.size(), cudaMemcpyDeviceToHost));
        for (unsigned i = 0; i < n; ++i) {
            unsigned kind = (i + scenario) % 8; bool native = mode == AOTX_POLICY_NATIVE;
            if (kind == 4) {
                aotx_check(!memcmp(&output[i], &original[i], sizeof(output[i])), "invalid rows preserve their output canary");
                aotx_check(!memcmp(after.data() + 16 * i, expected.data() + 16 * i, 16), "invalid rows preserve their state canary");
                continue;
            }
            uint64_t revision = (uint64_t)input[i].reserved1[1] | (uint64_t)input[i].reserved1[2] << 32;
            bool maintain = !native && kind == 5;
            bool appraise = abi == 2 && (kind == 0 || kind == 6 || (native ? kind == 5 : kind == 7));
            aotx_policy_output want = {};
            want.action = maintain ? AOTX_POLICY_MAINTAIN : appraise ? AOTX_POLICY_APPRAISE : AOTX_POLICY_QUIET;
            want.reason = maintain ? AOTX_POLICY_REASON_PRESSURE :
                appraise ? native ? input[i].source ^ revision : AOTX_POLICY_REASON_EVIDENCE : 0;
            aotx_check(!memcmp(&output[i], &want, sizeof(want)), "ABI, work, pressure and native selection produce the exact row proposal");
            aotx_check(aotx_get(after.data() + 16 * i) == 201 + i + (native ? input[i].reserved1[0] : 0),
                "each row advances its own native or data counter");
            aotx_check(aotx_get(after.data() + 16 * i + 8) ==
                (native ? revision : maintain ? input[i].source : 100 + i), "each row preserves its exact work or maintenance state");
        }
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    cudaFree(di); cudaFree(dout); cudaFree(db); cudaFree(da); aotx_policy_close();
    printf("policy appraisal batch N=%u ABI=%u mode=%u\n", n, abi, mode);
}
#endif
