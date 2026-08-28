/* Purpose: Check the record of a late verdict and the flag of a record made in a replay.
 * Owns: The buffers of those cases.
 * Threading: One host thread; the tool check calls the cases one at a time.
 * Lifetime: The program.
 *
 * The file is a part of the tool check. It reads the helpers of that check, so it comes
 * after them in the same translation unit. */
#ifndef AOTX_TEST_TOOL_VERDICT_H
#define AOTX_TEST_TOOL_VERDICT_H

/* The number of slots of the device ring the check reads back after a step. */
#define AOTX_TOOL_TEST_TAIL 512u

/* Write one note record of four bytes, with the replay mark as it stands. */
__global__ void aotx_tool_test_note(void)
{
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        const unsigned char text[4] = { 'f', 'l', 'a', 'g' };
        aotx_seam_write(AOTX_WRITER_SYSTEM, AOTX_CLASS_B, AOTX_REC_NOTE, 0u, text, 4u);
    }
}

/* Copies the last slots of the device ring to the host, in sequence order. Returns the
 * count copied, and fills the sequence of the first slot. */
static unsigned int aotx_tool_test_tail(aotx_record_header *out, unsigned long long *first)
{
    aotx_seam_state seam;
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    unsigned long long tail = seam.dev.tail;
    unsigned long long count = (tail < (unsigned long long)AOTX_TOOL_TEST_TAIL)
                             ? tail : (unsigned long long)AOTX_TOOL_TEST_TAIL;
    *first = tail - count + 1ull;
    for (unsigned long long i = 0ull; i < count; ++i) {
        unsigned long long seq = *first + i;
        const unsigned char *at = seam.dev.base
                                + ((seq - 1ull) & seam.dev.mask)
                                  * (unsigned long long)AOTX_SLOT_BYTES;
        aotx_check_runtime(cudaMemcpy((unsigned char *)out + i * AOTX_SLOT_BYTES, at,
                                      AOTX_SLOT_BYTES, cudaMemcpyDeviceToHost),
                           "cudaMemcpy");
    }
    return (unsigned int)count;
}

/* The state hash of the apply and the last claimed sequence of the device ring. */
static unsigned long long aotx_tool_test_hash(unsigned long long *tail)
{
    aotx_seam_state seam;
    aotx_check_runtime(cudaMemcpyFromSymbol(&seam, aotx_seam, sizeof seam),
                       "cudaMemcpyFromSymbol");
    *tail = seam.dev.tail;
    return seam.apply.state_hash;
}

/* The fold of one body into the state hash, as the device does it. */
static unsigned long long aotx_tool_test_fold(unsigned long long hash,
                                              const unsigned char *bytes, unsigned int count)
{
    for (unsigned int i = 0u; i < count; ++i) {
        hash ^= (unsigned long long)bytes[i];
        hash *= AOTX_FNV_PRIME;
    }
    return hash;
}

/* The record of a late verdict. A granted request whose deadline passed ends with a reply
 * record that the device writes. The record stands in the ring as class A, with the
 * writer of the requesting agent, the late status and one part. The state hash after the
 * step is the hash before it with the bodies of those records folded in, in sequence
 * order. A restore applies the record where it stands, so the fold is the proof that a
 * replay reaches the same hash. */
static void aotx_tool_test_case_verdict_record(aotx_pump *pump, unsigned int count,
                                               unsigned int *applied, unsigned int *failed)
{
    aotx_tool_test_clear<<<1, AOTX_SLOTS>>>(0u);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_tool_call *call =
        (aotx_tool_call *)calloc(AOTX_SLOTS, sizeof(aotx_tool_call));
    for (unsigned int i = 0u; i < count; ++i) {
        call[i].tool = AOTX_TOOL_FS_READ;
        call[i].arg_len = (unsigned int)snprintf(call[i].arg, AOTX_TOOL_ARG_BYTES,
                                                 "notes/verdict-%u.txt", i);
    }
    aotx_tool_call *on =
        (aotx_tool_call *)aotx_tool_test_take(AOTX_SLOTS * sizeof(aotx_tool_call));
    unsigned int *id =
        (unsigned int *)aotx_tool_test_take(AOTX_SLOTS * sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(on, call, AOTX_SLOTS * sizeof(aotx_tool_call),
                                  cudaMemcpyHostToDevice), "cudaMemcpy");
    unsigned long long tick = 0ull;
    aotx_check_runtime(cudaMemcpyFromSymbol(&tick, aotx_time_tick, sizeof tick),
                       "cudaMemcpyFromSymbol");
    aotx_tool_test_open<<<1, 1>>>(on, 0u, count, 1u, id, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int *ids = (unsigned int *)calloc(AOTX_SLOTS, sizeof(unsigned int));
    aotx_check_runtime(cudaMemcpy(ids, id, AOTX_SLOTS * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost), "cudaMemcpy");
    aotx_tool_test_auth_many<<<1, 1>>>(id, count, 1u, tick);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    /* Two ticks with the grant in place: the records of the grant leave the tail, so the
     * hash read below stands right before the verdict. */
    aotx_pump_tick(pump);
    aotx_pump_tick(pump);
    unsigned long long tail_before = 0ull;
    unsigned long long tail_after = 0ull;
    unsigned long long before = aotx_tool_test_hash(&tail_before);
    aotx_tool_test_expire_many<<<1, AOTX_SLOTS>>>(0u, count);
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_pump_tick(pump);
    unsigned long long after = aotx_tool_test_hash(&tail_after);

    aotx_record_header *ring =
        (aotx_record_header *)calloc(AOTX_TOOL_TEST_TAIL, AOTX_SLOT_BYTES);
    unsigned long long first = 0ull;
    unsigned int got = aotx_tool_test_tail(ring, &first);
    unsigned int found = 0u;
    unsigned int right = 0u;
    unsigned long long fold = before;
    unsigned long long last = 0ull;
    unsigned int in_order = 1u;
    for (unsigned int i = 0u; i < got; ++i) {
        const aotx_record_header *h =
            (const aotx_record_header *)((const unsigned char *)ring + i * AOTX_SLOT_BYTES);
        /* Only the records of the tick of the verdict count. An older verdict of another
         * case may still stand in the ring. */
        if (h->type != AOTX_REC_TOOL_REPLY || h->seq != first + i
            || h->seq <= tail_before || h->seq > tail_after) {
            continue;
        }
        const aotx_tool_reply_body *body =
            (const aotx_tool_reply_body *)((const unsigned char *)h + AOTX_HEADER_BYTES);
        if (body->status != AOTX_TOOL_LATE) {
            continue;
        }
        found += 1u;
        unsigned int slot = body->agent;
        if (h->cls == AOTX_CLASS_A && h->writer == AOTX_WRITER_AGENT_BASE + body->agent
            && body->parts == 1u && body->part == 0u && slot < count
            && body->request == ids[slot] && h->body_len == sizeof *body
            && (h->flags & AOTX_FLAG_REPLAY) == 0u) {
            right += 1u;
        }
        if (last != 0ull && body->agent <= last) {
            in_order = 0u;
        }
        last = body->agent;
        fold = aotx_tool_test_fold(fold, (const unsigned char *)body,
                                   (unsigned int)sizeof *body);
    }
    *applied += 1u;
    if (found != count || right != count || in_order == 0u) {
        printf("tool: %u of %u late verdicts stand in the ring as class A records of the "
               "requesting agent, %u right, in slot order %u\n", found, count, right,
               in_order);
        *failed += 1u;
    }
    *applied += 1u;
    if (after == before || after != fold) {
        printf("tool: the state hash after %u late verdicts is %016llx, before %016llx, "
               "the fold of their bodies gives %016llx\n", count, after, before, fold);
        *failed += 1u;
    }
    printf("tool: %u late verdicts stand in the ring as class A records and the state "
           "hash is the fold of their bodies\n", count);
    free(call);
    free(ids);
    free(ring);
    cudaFree(on);
    cudaFree(id);
}

/* The flag of a record made while a replay runs. The disk side reads it to derive no
 * request line that the feeder would execute a second time. */
static void aotx_tool_test_case_replay_flag(unsigned int *applied, unsigned int *failed)
{
    aotx_record_header *ring =
        (aotx_record_header *)calloc(AOTX_TOOL_TEST_TAIL, AOTX_SLOT_BYTES);
    unsigned long long first = 0ull;
    unsigned int flagged = 0u;
    unsigned int plain = 0u;
    aotx_seam_set_replaying(1);
    aotx_tool_test_note<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    aotx_seam_set_replaying(0);
    aotx_tool_test_note<<<1, 1>>>();
    aotx_check_runtime(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    unsigned int got = aotx_tool_test_tail(ring, &first);
    for (unsigned int i = 0u; i < got; ++i) {
        const aotx_record_header *h =
            (const aotx_record_header *)((const unsigned char *)ring + i * AOTX_SLOT_BYTES);
        if (h->type != AOTX_REC_NOTE || h->seq != first + i || h->body_len != 4u) {
            continue;
        }
        const unsigned char *text = (const unsigned char *)h + AOTX_HEADER_BYTES;
        if (memcmp(text, "flag", 4u) != 0) {
            continue;
        }
        if ((h->flags & AOTX_FLAG_REPLAY) != 0u) {
            flagged += 1u;
        } else {
            plain += 1u;
        }
    }
    *applied += 1u;
    if (flagged != 1u || plain != 1u) {
        printf("tool: %u records made in a replay carry the replay flag and %u made "
               "outside it carry none; one of each is expected\n", flagged, plain);
        *failed += 1u;
    }
    printf("tool: a record made while a replay runs carries the replay flag and a record "
           "made outside it carries none\n");
    free(ring);
}

#endif
