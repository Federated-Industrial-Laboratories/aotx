/* Purpose: Provide live GPU state and a real checkpoint transport for memory tests.
 * Owns: Runtime setup, mapped ring lifetime and exact snapshot byte access.
 * Launch shape: One profile batch and the normal checkpoint graph nodes.
 * Lifetime: One test scope; each fixture releases its transport and input buffers. */
#ifndef AOTX_CHECKPOINT_FIXTURE_H
#define AOTX_CHECKPOINT_FIXTURE_H
#include "live_fixture.h"
#include "cognitive/checkpoint.cuh"
#include "cognitive/checkpoint_io.h"
#ifdef AOTX_AFFECT
#include "quality/quality.cuh"
#endif
#include <unistd.h>

__global__ void aotx_checkpoint_test_clear(void) {
    unsigned i = threadIdx.x;
    if (i < AOTX_SLOTS) {
        aotx_seqs.slot[i].state = AOTX_SEQ_STATE_FREE;
        aotx_tool_embed.state[i] = AOTX_TOOL_EMBED_NONE;
#ifdef AOTX_AFFECT
        aotx_quality_state[i] = {};
#endif
    }
}
struct aotx_checkpoint_device {
    aotx_live_device live;
    aotx_seam_rings transport = {};
    aotx_checkpoint_device(unsigned n) : live(n) {
        AOTX_LIVE_CLEAR(aotx_checkpoint);
        auto seam = live.seam(); seam.boot_id = 0x12970000 + n;
        AOTX_CUDA(cudaMemcpyToSymbol(aotx_seam, &seam, sizeof(seam)));
        aotx_checkpoint_test_clear<<<1,AOTX_SLOTS>>>();
        aotx_check(!aotx_checkpoint_open(&transport, seam.boot_id), "checkpoint transport opens");
    }
    ~aotx_checkpoint_device() { AOTX_LIVE_CLEAR(aotx_checkpoint); aotx_checkpoint_close(&transport); }
    aotx_checkpoint_ring *ring() { return (aotx_checkpoint_ring *)transport.checkpoint_map; }
    aotx_checkpoint_state state() {
        aotx_checkpoint_state s; AOTX_CUDA(cudaMemcpyFromSymbol(&s, aotx_checkpoint, sizeof(s))); return s;
    }
    void step() { aotx_checkpoint_capture(nullptr); AOTX_CUDA(cudaDeviceSynchronize()); }
    void publish(uint64_t target) {
        unsigned limit = AOTX_CP_BYTES / AOTX_CP_COPY + 4;
        for (unsigned i = 0; i < limit && ring()->head < target; ++i) step();
        aotx_check(ring()->head == target, "checkpoint publishes within its copy bound");
    }
    aotx_bytes image(uint64_t serial) {
        const unsigned char *slot = (const unsigned char *)(ring() + 1) +
            ((serial - 1) % AOTX_MEMORY_SNAPSHOTS) * AOTX_CP_SLOT_BYTES;
        aotx_check(aotx_get(slot + 8) == serial && aotx_get(slot) == ring()->boot,
            "slot identity names its exact boot and serial");
        uint64_t bytes = aotx_get(slot + 16);
        aotx_check(bytes <= AOTX_CP_BYTES, "snapshot length fits the configured transport");
        return aotx_bytes(slot + AOTX_CP_SLOT_HEADER, slot + AOTX_CP_SLOT_HEADER + bytes);
    }
};
#endif
