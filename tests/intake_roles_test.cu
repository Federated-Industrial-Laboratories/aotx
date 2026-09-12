/* Purpose: Check recorded interpretation ownership with multiple resident language models.
 * Owns: Distinct sources, exact decisions and unavailable or changed owner controls.
 * Launch shape: N=1 and N=64 through live admission and journal recovery.
 * Lifetime: One test process; model response bytes use the maintained intake fixture. */
#include "intake_fixture.h"

__global__ void aotx_intake_roles_setup(unsigned mode)
{
    if (threadIdx.x) return;
    aotx_model[AOTX_MODEL_LANGUAGE_AUDIO].layers = 1;
    aotx_model_wrap[AOTX_MODEL_LANGUAGE_AUDIO] = aotx_model_wrap[AOTX_MODEL_LANGUAGE];
    auto &audio = aotx_model_load.resident[AOTX_MODEL_LANGUAGE_AUDIO];
    audio.active = mode < 1 || mode == 3; audio.slot = AOTX_MODEL_LANGUAGE_AUDIO;
    for (unsigned j = 0; j < 32; ++j) audio.body.digest[j] = 0xa1 + j;
    if (mode == 2) {
        auto &embed = aotx_model_load.resident[AOTX_MODEL_EMBEDDING];
        embed.active = 1;
        for (unsigned j = 0; j < 32; ++j) embed.body.digest[j] = audio.body.digest[j];
    }
    if (mode == 3) audio.body.digest[31] ^= 1;
}
static void aotx_intake_roles_roundtrip(unsigned n, bool mixed)
{
    aotx_live_records start, records;
    aotx_bytes expected, query, choice;
    {
        aotx_intake_device d(n); aotx_fixture empty;
        aotx_intake_roles_setup<<<1,1>>>(0); AOTX_CUDA(cudaDeviceSynchronize());
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), AOTX_LIVE_LOAD);
        auto binding = d.send(aotx_intake_bind(n), AOTX_LIVE_BIND);
        start.insert(start.end(), binding.begin(), binding.end());
        query = aotx_intake_query(n, 0, 1);
        for (unsigned i = 0; i < n; ++i) if (mixed && (n == 1 || i % 2 == 0)) {
            auto *q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            const char hex[] = "0123456789abcdef";
            std::string source = "[audio:" + std::string(62, 'a') + hex[i / 16] + hex[i % 16]
                + "] Iris" + std::to_string(i) + " will cook.";
            memset(q + 4640, 0, AOTX_RECALL_TEXT);
            memcpy(q + 4640, source.data(), source.size()); aotx_put(q + 148, source.size(), 4);
        }
        records = d.intake(query, aotx_intake_initial(n));
        choice = aotx_retain_result(records, AOTX_INTAKE_CHOICE);
        aotx_check(!d.state().status && choice.size() > 64, "live interpretation decision succeeds");
        if (d.state().status || choice.size() <= 64) return;
        for (unsigned i = 0; i < n; ++i) {
            bool audio = mixed && (n == 1 || i % 2 == 0);
            auto *meta = choice.data() + 64 + i * AOTX_LIVE_INTAKE_ROW + AOTX_LIVE_AUTO_ROW;
            for (unsigned j = 0; j < 32; ++j)
                aotx_check(meta[8 + j] == (audio ? 0xa1u : 0x51u) + j, "selected parent digest is exact");
            aotx_check(aotx_get(meta + 72, 4) == 3, "each source has three nonempty quotations");
        }
        expected = aotx_retain_store();
    }
    {
        aotx_intake_device d(n);
        aotx_intake_roles_setup<<<1,1>>>(0); AOTX_CUDA(cudaDeviceSynchronize());
        d.process(start, true); d.process(records, true);
        aotx_check(!d.state().fatal && d.state().replays == n, "each unchanged decision recovers");
        aotx_check(aotx_retain_store() == expected, "mixed model recovery preserves every store byte");
    }
    if (!mixed) return;
    for (unsigned control = 0; control < 6; ++control) {
        aotx_intake_device d(n);
        unsigned setup = control < 3 ? control + 1 : 0;
        aotx_intake_roles_setup<<<1,1>>>(setup); AOTX_CUDA(cudaDeviceSynchronize());
        d.process(start, true);
        auto before = aotx_retain_store();
        d.process(aotx_live_parts(query, AOTX_LIVE_QUERY, 3), true);
        auto changed = choice;
        auto *model = changed.data() + 64 + AOTX_LIVE_AUTO_ROW + 8;
        if (control == 3) for (unsigned j = 0; j < 32; ++j) model[j] = 0x51 + j;
        if (control == 4) memset(model, 0, 32);
        if (control == 5) model[31] ^= 1;
        d.process(aotx_live_parts(changed, AOTX_INTAKE_CHOICE, 3), true);
        aotx_check(d.state().fatal && !d.state().replays, "unavailable or changed owner refuses the complete batch");
        aotx_check(aotx_retain_store() == before, "refused ownership publishes no store change");
        for (const auto &binding : d.bindings(n))
            aotx_check(!binding.ordinal, "refused ownership publishes no binding change");
    }
}
int main(void)
{
    int cards = 0; if (cudaGetDeviceCount(&cards) != cudaSuccess || !cards) return 77;
    for (unsigned n : {1u, 64u}) {
        aotx_intake_roles_roundtrip(n, false); aotx_intake_roles_roundtrip(n, true);
        printf("intake roles N=%u checks=%u failures=%u\n", n, aotx_checks, aotx_failures);
    }
    return aotx_failures ? 1 : 0;
}
