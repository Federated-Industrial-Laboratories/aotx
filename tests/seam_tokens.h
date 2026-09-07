/* Purpose: Check token application refusals through the inbound ring.
 * Owns: Distinct token records and sequence-open refusal fixtures.
 * Launch shape: One thread per slot, at one and the profile slot count.
 * Lifetime: The final case sets of one seam test. */
#ifndef AOTX_TEST_SEAM_TOKENS_H
#define AOTX_TEST_SEAM_TOKENS_H

__global__ void aotx_test_token_reset(void)
{
    unsigned int slot = threadIdx.x;
    if (slot >= AOTX_SLOTS) return;
    memset(&aotx_seqs.slot[slot], 0, sizeof aotx_seqs.slot[slot]);
    aotx_seqs.tokens[slot][0] = -1 - (int)slot;
    if (slot == 0u) {
        aotx_seqs.live = 0u;
        aotx_seqs.refused = 0u;
    }
}

__global__ void aotx_test_token_read(unsigned int *rows, unsigned int count)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    rows[slot] = aotx_seqs.tokens[slot][0] == (int)(1000u + slot)
        && aotx_seqs.slot[slot].prompt == 1u
        && aotx_seqs.slot[slot].state == AOTX_SEQ_STATE_DONE;
}

__global__ void aotx_test_open_refuse(unsigned int count)
{
    unsigned int slot = threadIdx.x;
    if (slot >= count) return;
    int token = (int)(3000u + slot);
    aotx_model_how sample = {};
    aotx_seq_open(slot, AOTX_MODEL_LANGUAGE, &token, 1u, 1u, 0u, &sample, 0ull);
}

static void aotx_test_token_check(int good, const char *name, unsigned int count,
                                  unsigned int *applied, unsigned int *failed)
{
    *applied += 1u;
    if (!good) {
        printf("seam: FAIL %s at %u slots\n", name, count);
        *failed += 1u;
    }
}

static void aotx_test_token_apply(aotx_pump *pump, const aotx_seam_rings *rings,
                                  aotx_test_consumer *state, unsigned long long boot_id,
                                  unsigned long long *fed, unsigned int count,
                                  unsigned int *applied, unsigned int *failed)
{
    unsigned int rows[AOTX_SLOTS], *device = NULL;
    aotx_check_runtime(cudaMalloc(&device, count * sizeof *device), "cudaMalloc");
    aotx_test_token_reset<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_seam_set_replaying(1);
    for (unsigned int bad = 0u; bad < 2u; ++bad) {
        aotx_pump_report before, after;
        aotx_pump_read(&before);
        for (unsigned int slot = 0u; slot < count; ++slot) {
            aotx_token_body body = {};
            body.slot = slot;
            body.role = AOTX_MODEL_LANGUAGE;
            body.position = bad ? slot + 2u : 0u;
            body.token = (bad ? 2000u : 1000u) + slot;
            body.seed = 7ull + slot;
            body.flags = AOTX_TOKEN_PROMPT | AOTX_TOKEN_LAST;
            aotx_test_put(rings, boot_id, *fed, AOTX_WRITER_RESTORE, AOTX_CLASS_A,
                          AOTX_REC_TOKEN, AOTX_FLAG_REPLAYED, &body, sizeof body);
            *fed += 1ull;
        }
        aotx_pump_tick(pump);
        aotx_test_settle(state, rings);
        aotx_pump_read(&after);
        aotx_test_token_check(after.rejected == before.rejected + (bad ? count : 0u),
            bad ? "failed token application is reported" : "valid tokens are accepted",
            count, applied, failed);
        aotx_test_token_check(after.refused == before.refused + (bad ? count : 0u),
            "decode refusal count matches token acceptance", count, applied, failed);
        aotx_test_token_read<<<1, AOTX_SLOTS>>>(device, count);
        aotx_check_runtime(cudaMemcpy(rows, device, count * sizeof *device,
                                       cudaMemcpyDeviceToHost), "cudaMemcpy");
        for (unsigned int slot = 0u; slot < count; ++slot) {
            aotx_test_token_check(rows[slot], "each slot retains its accepted token",
                                  count, applied, failed);
        }
        printf("seam: %u %s tokens, %llu rejected, %u decode refusals\n", count,
               bad ? "invalid" : "valid", after.rejected - before.rejected,
               after.refused - before.refused);
    }
    aotx_pump_report before, after;
    aotx_pump_read(&before);
    aotx_test_open_refuse<<<1, AOTX_SLOTS>>>(count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_pump_read(&after);
    aotx_test_token_check(after.refused == before.refused + count,
                          "ordinary opens can be refused", count, applied, failed);
    aotx_test_token_check(after.rejected == before.rejected,
                          "ordinary open refusals reject no input record", count, applied, failed);
    printf("seam: %u ordinary open refusals, %llu rejected input records\n",
           after.refused - before.refused, after.rejected - before.rejected);
    aotx_test_token_reset<<<1, AOTX_SLOTS>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_seam_set_replaying(0);
    cudaFree(device);
}

#endif
