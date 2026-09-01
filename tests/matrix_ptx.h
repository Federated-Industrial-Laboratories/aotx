/* Purpose: Check the hand-written GEMV module against the compiled kernel.
 * Owns: The module fixture.
 * Launch shape: One driver graph node for the assembly module.
 * Lifetime: One matrix test case. */

/* The module of hand written assembly, as a node of a graph through the driver. The result
 * and the rate both go against the kernel the compiler makes. */
static unsigned int aotx_test_case_ptx(unsigned int *applied, unsigned int *skipped)
{
    char *text = aotx_test_module(AOTX_PTX_DIR "/gemv_q8.ptx");
    if (text == NULL) {
        printf("matrix: skipped 2 module cases; the module text did not read\n");
        *skipped += 2u;
        return 0u;
    }
    CUmodule module;
    CUfunction function;
    aotx_check_driver(cuModuleLoadData(&module, text), "cuModuleLoadData");
    free(text);
    aotx_check_driver(cuModuleGetFunction(&function, module, "aotx_gemv_q8"),
                      "cuModuleGetFunction");

    unsigned int n = 2560u;
    unsigned int k = 2560u;
    unsigned int m = 1u;
    aotx_test_tensor w;
    aotx_test_build(&w, AOTX_TENSOR_Q8_0, n, k, AOTX_TEST_SEED + 200u);
    half *dx = NULL;
    half *x = aotx_test_input(m, k, AOTX_TEST_SEED + 201u, &dx);
    float *dy = NULL;
    float *dz = NULL;
    aotx_check_runtime(cudaMalloc((void **)&dy, (size_t)n * sizeof *dy), "cudaMalloc");
    aotx_check_runtime(cudaMalloc((void **)&dz, (size_t)n * sizeof *dz), "cudaMalloc");
    aotx_check_runtime(cudaMemset(dz, 0, (size_t)n * sizeof *dz), "cudaMemset");

    CUdeviceptr pw = (CUdeviceptr)w.device;
    CUdeviceptr px = (CUdeviceptr)dx;
    CUdeviceptr pz = (CUdeviceptr)dz;
    CUdeviceptr pm = 0;
    void *params[] = { &pw, &n, &k, &px, &pz, &pm };
    CUDA_KERNEL_NODE_PARAMS node_params = {};
    node_params.func = function;
    node_params.gridDimX = (n + AOTX_GEMV_ROWS_CTA - 1u) / AOTX_GEMV_ROWS_CTA;
    node_params.gridDimY = 1u;
    node_params.gridDimZ = 1u;
    node_params.blockDimX = AOTX_GEMV_THREADS;
    node_params.blockDimY = 1u;
    node_params.blockDimZ = 1u;
    node_params.kernelParams = params;

    CUgraph graph;
    CUgraphExec exec;
    CUgraphNode node;
    aotx_check_driver(cuGraphCreate(&graph, 0), "cuGraphCreate");
    aotx_check_driver(cuGraphAddKernelNode(&node, graph, NULL, 0, &node_params),
                      "cuGraphAddKernelNode");
    aotx_check_driver(cuGraphInstantiate(&exec, graph, 0), "cuGraphInstantiate");

    for (unsigned int i = 0u; i < aotx_test_warm; ++i) {
        aotx_check_driver(cuGraphLaunch(exec, 0), "cuGraphLaunch");
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    double from = aotx_test_now();
    for (unsigned int i = 0u; i < aotx_test_runs; ++i) {
        aotx_check_driver(cuGraphLaunch(exec, 0), "cuGraphLaunch");
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double raw = (aotx_test_now() - from) / (double)aotx_test_runs;

    from = aotx_test_now();
    for (unsigned int i = 0u; i < aotx_test_runs; ++i) {
        aotx_test_gemv(&w, dx, m, dy);
    }
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    double made = (aotx_test_now() - from) / (double)aotx_test_runs;

    float *a = (float *)malloc((size_t)n * sizeof *a);
    float *b = (float *)malloc((size_t)n * sizeof *b);
    aotx_check_runtime(cudaMemcpy(a, dy, (size_t)n * sizeof *a, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    aotx_check_runtime(cudaMemcpy(b, dz, (size_t)n * sizeof *b, cudaMemcpyDeviceToHost),
                       "cudaMemcpy");
    double *want = (double *)malloc((size_t)n * sizeof *want);
    aotx_test_ref(&w, x, m, want, 0);
    double one = 0.0;
    double two = 0.0;
    unsigned int wrong = aotx_test_diff(b, want, n, AOTX_TEST_TOL, &one);
    unsigned int bad = aotx_test_diff(a, want, n, AOTX_TEST_TOL, &two);
    double rate_raw = (double)w.bytes / raw / 1e9;
    double rate_made = (double)w.bytes / made / 1e9;
    printf("matrix: the module gives %u of %u sums over %.0e of the reference, worst %.2e; "
           "the kernel gives %u, worst %.2e\n", wrong, n, AOTX_TEST_TOL, one, bad, two);
    printf("matrix: the module runs at %.1f GB a second and the kernel at %.1f, "
           "%.0f us against %.0f us\n", rate_raw, rate_made, raw * 1e6, made * 1e6);
    *applied += (aotx_test_lowered != 0) ? 1u : 2u;
    aotx_test_left_out += (aotx_test_lowered != 0) ? 1u : 0u;
    unsigned int failed = (wrong != 0u || bad != 0u) ? 1u : 0u;
    free(want);
    if (aotx_test_lowered == 0 && rate_raw < AOTX_TEST_GEMV_ONE) {
        printf("matrix: the gemv rate gate refuses the module at %.1f GB a second\n",
               rate_raw);
        failed += 1u;
    }

    free(a);
    free(b);
    free(x);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    aotx_test_free(&w);
    aotx_check_driver(cuGraphExecDestroy(exec), "cuGraphExecDestroy");
    aotx_check_driver(cuGraphDestroy(graph), "cuGraphDestroy");
    aotx_check_driver(cuModuleUnload(module), "cuModuleUnload");
    return failed;
}
