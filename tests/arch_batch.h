/* Purpose: Compare distinct sequences through prefill and the decode child graph.
 * Owns: The arrays and graph instances of the comparison.
 * Threading: One host thread; device calls take all selected slots.
 * Lifetime: One model check. */
#ifndef AOTX_TEST_ARCH_BATCH_H
#define AOTX_TEST_ARCH_BATCH_H
#include "model/decode.cuh"
#include "model/graph_host.h"

/* Keep the actual child graph. The parent plan and commit need the tick's journal rings. */
static cudaGraphExec_t aotx_arch_decode_graph(void)
{
    if (aotx_decode_open() != 0) return NULL;
    cudaStream_t stream;
    cudaGraph_t graph = NULL;
    cudaGraphExec_t exec = NULL;
    aotx_check_runtime(cudaStreamCreate(&stream), "cudaStreamCreate");
    aotx_check_runtime(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                       "cudaStreamBeginCapture");
    int bad = aotx_decode_capture(stream);
    aotx_check_runtime(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    size_t count = 0u;
    aotx_check_runtime(cudaGraphGetNodes(graph, NULL, &count), "cudaGraphGetNodes");
    cudaGraphNode_t *nodes = (cudaGraphNode_t *)calloc(count, sizeof *nodes);
    unsigned int children = 0u;
    if (nodes != NULL && bad == 0) {
        aotx_check_runtime(cudaGraphGetNodes(graph, nodes, &count), "cudaGraphGetNodes");
        for (size_t i = 0u; i < count; ++i) {
            cudaGraphNodeType type;
            aotx_check_runtime(cudaGraphNodeGetType(nodes[i], &type), "cudaGraphNodeGetType");
            if (type != cudaGraphNodeTypeGraph) continue;
            cudaGraph_t child;
            children += 1u;
            aotx_check_runtime(cudaGraphChildGraphNodeGetGraph(nodes[i], &child),
                               "cudaGraphChildGraphNodeGetGraph");
            if (children == 1u) {
                cudaGraph_t copy;
                aotx_check_runtime(cudaGraphClone(&copy, child), "cudaGraphClone");
                aotx_check_runtime(cudaGraphInstantiate(&exec, copy, 0), "cudaGraphInstantiate");
                cudaGraphDestroy(copy);
            }
        }
    }
    free(nodes);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    if (children != 1u && exec != NULL) { cudaGraphExecDestroy(exec); exec = NULL; }
    return exec;
}

static void aotx_arch_batch(aotx_arch_gear *gear, aotx_kv_map *map, unsigned int role,
                             const aotx_arch_list *list, const char *file, unsigned int vocab)
{
    constexpr unsigned int capacity = AOTX_MODEL_MAX_TOKENS / AOTX_SLOTS;
    constexpr unsigned int width = capacity < 5u ? capacity : 5u;
    static_assert(width > 0u && width * AOTX_SLOTS <= AOTX_ARCH_IDS, "the batch must fit");
    const unsigned int steps = 3u;
    int ids[AOTX_SLOTS * width], token[AOTX_SLOTS];
    int serial[AOTX_SLOTS][3], batched[AOTX_SLOTS][3];
    unsigned int offsets[AOTX_SLOTS + 1u], agents[AOTX_SLOTS];
    aotx_model_how how[AOTX_SLOTS];
    aotx_model_how *device_how = NULL;
    memset(how, 0, sizeof how);
    for (unsigned int s = 0u; s < AOTX_SLOTS; ++s) {
        how[s].top_k = 1u; how[s].top_p = 1.0f; how[s].repeat_penalty = 1.0f;
        how[s].think_limit = -1;
        for (unsigned int k = 0u; k < AOTX_MODEL_STEERS; ++k)
            how[s].steer[k] = AOTX_MODEL_CONDUCT_NONE;
        how[s].voice = AOTX_MODEL_CONDUCT_NONE;
    }
    aotx_check_runtime(cudaMalloc((void **)&device_how, sizeof how), "cudaMalloc");
    aotx_check_runtime(cudaMemcpy(device_how, how, sizeof how, cudaMemcpyHostToDevice),
                       "cudaMemcpy");
    cudaGraphExec_t decode = aotx_arch_decode_graph();
    int bad = decode == NULL || vocab <= 1000u + AOTX_SLOTS || list->prefills == 0u;
    unsigned int before = aotx_arch_mapped();
    double elapsed[2][2] = {};
    for (unsigned int pass = 0u; pass < 2u && !bad; ++pass) {
        unsigned int groups = pass == 0u ? AOTX_SLOTS : 1u;
        unsigned int seqs = pass == 0u ? 1u : AOTX_SLOTS;
        for (unsigned int group = 0u; group < groups && !bad; ++group) {
            aotx_model_forget();
            for (unsigned int s = 0u; s < seqs; ++s) {
                unsigned int slot = pass == 0u ? group : s;
                agents[s] = slot;
                for (unsigned int k = 0u; k < width - 1u; ++k)
                    ids[s * width + k] = (int)list->prefill[k % list->prefills];
                ids[s * width + width - 1u] = (int)(1000u + slot);
            }
            for (unsigned int step = 0u; step < steps && !bad; ++step) {
                struct timespec began, ended;
                clock_gettime(CLOCK_MONOTONIC, &began);
                unsigned int tokens = step == 0u ? width : 1u;
                for (unsigned int s = 0u; s <= seqs; ++s) offsets[s] = s * tokens;
                aotx_check_runtime(cudaMemcpy(gear->ids, ids, seqs * tokens * sizeof(int),
                                              cudaMemcpyHostToDevice), "cudaMemcpy");
                aotx_check_runtime(cudaMemcpy(gear->offset, offsets, (seqs + 1u) * sizeof(unsigned int),
                                              cudaMemcpyHostToDevice), "cudaMemcpy");
                aotx_check_runtime(cudaMemcpy(gear->agent, agents, seqs * sizeof(unsigned int),
                                              cudaMemcpyHostToDevice), "cudaMemcpy");
                bad = aotx_model_pages(role, gear->offset, seqs, gear->agent) != 0;
                aotx_kv_serve(map, 0);
                if (bad) break;
                if (step == 0u) {
                    unsigned long long seed = 0ull;
                    bad = aotx_model_sample(role, gear->ids, gear->offset, seqs, gear->agent,
                                            how, gear->token, NULL, &seed) != 0;
                } else {
                    aotx_model_run run = {};
                    run.ids = gear->ids; run.offset = gear->offset; run.agent = gear->agent;
                    run.token = gear->token; run.how = device_how;
                    run.seqs = seqs; run.tokens = seqs; run.rows = seqs;
                    run.select = AOTX_MODEL_ROWS_LAST;
                    aotx_check_runtime(cudaMemcpyToSymbol(aotx_model_call, &run, sizeof run,
                                                          (size_t)role * sizeof run),
                                       "cudaMemcpyToSymbol");
                    aotx_check_runtime(cudaGraphLaunch(decode, 0), "cudaGraphLaunch");
                }
                aotx_check_runtime(cudaMemcpy(token, gear->token, seqs * sizeof(int),
                                              cudaMemcpyDeviceToHost), "cudaMemcpy");
                bad |= aotx_model_faulted() != 0u;
                clock_gettime(CLOCK_MONOTONIC, &ended);
                elapsed[pass][step == 0u ? 0u : 1u] += (double)(ended.tv_sec - began.tv_sec)
                    + (double)(ended.tv_nsec - began.tv_nsec) / 1.0e9;
                for (unsigned int s = 0u; s < seqs; ++s) {
                    unsigned int slot = agents[s];
                    (pass == 0u ? serial : batched)[slot][step] = token[s];
                    ids[s] = token[s];
                }
            }
            for (unsigned int s = 0u; s < seqs; ++s) aotx_arch_release<<<1, 1>>>(agents[s]);
            aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            aotx_kv_serve(map, 0);
        }
    }
    for (unsigned int s = 0u; s < AOTX_SLOTS && !bad; ++s) {
        for (unsigned int step = 0u; step < steps; ++step) {
            if (serial[s][step] == batched[s][step]) continue;
            printf("arch: batch first difference slot %u step %u serial %d batch %d\n",
                   s, step, serial[s][step], batched[s][step]);
            bad = 1;
            break;
        }
    }
    printf("arch: batch %u distinct prompts, one prefill and two decode child passes\n",
           (unsigned int)AOTX_SLOTS);
    printf("arch: serial prefill %.3f s decode %.3f s; batch prefill %.3f s decode %.3f s\n",
           elapsed[0][0], elapsed[0][1], elapsed[1][0], elapsed[1][1]);
    aotx_arch_check(!bad && aotx_arch_mapped() == before, file,
                    "distinct batch matches serial prefill and decode child graph");
    if (decode != NULL) cudaGraphExecDestroy(decode);
    aotx_decode_close();
    cudaFree(device_how);
}
#endif
