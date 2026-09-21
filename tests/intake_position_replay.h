/* Purpose: Verify repeated positional payloads and token-independent recorded bytes.
 * Owns: Full choice recovery and an identical-text offset mutation.
 * Launch shape: N=1 and N=64 through combined admission and replay.
 * Lifetime: One maintained fixture without decoder execution. */
#ifndef AOTX_TEST_INTAKE_POSITION_REPLAY_H
#define AOTX_TEST_INTAKE_POSITION_REPLAY_H
static void aotx_stage_position_replay(unsigned n, bool optional) {
    aotx_live_records start, records; aotx_bytes expected, choice;
    {
        aotx_intake_device d(n); aotx_fixture empty;
        start = d.send(aotx_live_load_bytes(empty.wire(false, 0)), 1);
        auto binding = d.send(aotx_intake_bind(n), 3); start.insert(start.end(), binding.begin(), binding.end());
        auto query = aotx_stage_query(n); std::vector<std::string> replies;
        for (unsigned i = 0; i < n; ++i) {
            auto quote = optional ? aotx_source_unit(i) : "Echo" + std::to_string(i) + "!";
            auto source = optional ? quote : quote + " " + quote;
            auto q = query.data() + 128 + i * AOTX_LIVE_QUERY_ROW;
            memset(q + 4640, 0, 2048); memcpy(q + 4640, source.data(), source.size()); aotx_put(q + 148, source.size(), 4);
            auto first = aotx_stage_label(3, quote), final = aotx_stage_item(3, quote);
            aotx_intake_first_outputs.push_back("[" + first + (optional ? "" : "," + first) + "]");
            replies.push_back("[" + final + (optional ? "," + aotx_stage_item(1 + i % 2, quote) + "," + aotx_stage_item(2 - i % 2, quote) : "," + final) + "]");
        }
        records = d.intake(query, replies); expected = aotx_retain_store();
        aotx_check(!d.state().status, "positional required and unique optional quotes admit as one choice");
        if (d.state().status) return;
        choice = aotx_retain_result(records, 14);
    }
    {
        aotx_intake_device d(n); d.process(start, true); d.process(records, true);
        aotx_check(!d.state().fatal && aotx_retain_store() == expected,
            "recorded repeated positions and optional arrays recover independently of token splits");
    }
    if (optional) return;
    size_t tail = 64 + n * AOTX_LIVE_INTAKE_SOURCE_ROW;
    unsigned objects = aotx_get(choice.data() + tail + 20, 4);
    unsigned object = 3 * n + (n - 1) * 2 + 1;
    auto descriptor = choice.data() + tail + AOTX_COG_HEADER + object * AOTX_COG_OBJECT;
    size_t offset = tail + AOTX_COG_HEADER + objects * AOTX_COG_OBJECT + aotx_get(descriptor + AOTX_CO_OFFSET) + 20;
    aotx_check(choice[offset] != 0, "the second identical quote has a distinct original offset");
    aotx_intake_device d(n); d.process(start, true); auto before = aotx_retain_store(); auto bad = records;
    aotx_stage_mutate(bad, offset, choice[offset]); d.process(bad, true);
    aotx_check(d.state().fatal && aotx_retain_store() == before,
        "moving identical text to the first occurrence refuses canonical replay");
}
#endif
