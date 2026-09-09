// Purpose: Check typed tool status, forged console isolation, and bounded offline queries.
// Owns: Temporary disk replicas and the real policy stream writer for each fixture.
// Launch shape: One host process checks batches of one and 64 conversations.
// Lifetime: Each fixture removes its temporary replica before return.
#include "tool_fix.hpp"
#include "replica/replica.hpp"
#include "replica/schema.hpp"
#include "tools/query.hpp"
#include "cuda/tool/policy.h"
extern "C" {
#include "disk/drain/tool_policy.h"
}
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {
std::string policy_line(unsigned agent, unsigned defaults, unsigned choices,
                         unsigned selected, unsigned effective)
{
    return "{\"tick\":1,\"seq\":1,\"agent\":" + std::to_string(agent)
        + ",\"defaults\":" + std::to_string(defaults) + ",\"choices\":" + std::to_string(choices)
        + ",\"selected\":" + std::to_string(selected) + ",\"effective\":" + std::to_string(effective) + "}";
}
struct Record {
    aotx_record_header header{};
    aotx_tool_policy_body body{};
};
Record policy_record(unsigned agent, unsigned sequence, unsigned defaults, unsigned choices,
                      unsigned selected, unsigned effective)
{
    Record record;
    record.header.magic = AOTX_WIRE_MAGIC;
    record.header.layout = AOTX_WIRE_LAYOUT;
    record.header.header_bytes = AOTX_HEADER_BYTES;
    record.header.type = AOTX_REC_TOOL_POLICY;
    record.header.cls = AOTX_CLASS_B;
    record.header.writer = AOTX_WRITER_SYSTEM;
    record.header.body_len = sizeof record.body;
    record.header.tick = 1u;
    record.header.seq = sequence;
    record.body = {agent, defaults, choices, selected, effective};
    return record;
}
}

void aotx_ctrl_tool_fix(int &applied, int &failed)
{
    const auto check = [&](bool value, const char *reason) {
        ++applied;
        if (!value) { ++failed; std::printf("ctrl tools: %s\n", reason); }
    };
    using namespace aotx::ctrl;
    using namespace aotx::ctrl::replica;
    ToolPolicy policy;
    check(schema::tool_policy(policy_line(1u, 0u, 2u, 1u, 1u), policy)
        && policy.agent == 1u && policy.choices == 2u, "enabled override was not parsed");
    check(schema::tool_policy(policy_line(2u, 1023u, 1u, 1022u, 6u), policy),
        "disabled override was not parsed");
    for (const auto &line : {
        policy_line(1u, 0u, 3u, 0u, 0u), policy_line(1u, 0u, 0u, 1u, 1u),
        policy_line(1u, 0u, 0u, 0u, 1u), policy_line(256u, 0u, 0u, 0u, 0u),
        policy_line(1u, 1024u, 0u, 0u, 0u), policy_line(1u, 0u, 0u, 0u, 0u) + "extra",
        std::string("tools: agent 1 defaults 0 choices 0 selected 0 effective 0"),
        std::string("{\"tick\":0,\"seq\":0,\"agent\":0,\"defaults\":0,\"choices\":0,\"selected\":0,\"effective\":0}"),
        std::string("{\"tick\":1,\"seq\":1,\"agent\":0,\"defaults\":0,\"choices\":0,\"selected\":0,\"effective\":0,\"extra\":1}"),
        std::string("{\"tick\":1,\"seq\":1,\"agent\":0,\"defaults\":0,\"choices\":0,\"selected\":0,\"agent\":1}"),
        std::string("{\"tick\":1,\"seq\":1,\"agent\":0.5,\"defaults\":0,\"choices\":0,\"selected\":0,\"effective\":0}")})
        check(!schema::tool_policy(line, policy), "an invalid status was accepted");
    for (unsigned count : {1u, 64u}) {
        std::array<char, 40> name{};
        std::snprintf(name.data(), name.size(), "/tmp/aotx_tool_fix_XXXXXX");
        const char *made = mkdtemp(name.data());
        check(made != nullptr, "the fixture directory did not open");
        if (made == nullptr) continue;
        const std::filesystem::path root(made);
        const auto boot = root / "0000000000000001";
        std::filesystem::create_directories(boot / "transcript");
        std::filesystem::create_directories(root / "bus");
        std::ofstream(root / "phase") << "closed 1\n";
        std::ofstream(boot / "console.log");
        aotx_tool_policy_stream *stream = nullptr;
        check(aotx_tool_policy_stream_open(&stream, boot.c_str()) == 0, "the policy stream did not open");
        if (stream == nullptr) { std::filesystem::remove_all(root); continue; }
        for (unsigned agent = 0u; agent < count; ++agent) {
            unsigned group = agent % AOTX_TOOL_POLICY_GROUPS;
            unsigned selected = 1u << group;
            unsigned choices = 2u << (group * 2u);
            Record record = policy_record(agent, agent + 1u, 0u, choices, selected, selected);
            check(aotx_tool_policy_stream_record(stream, &record.header) == 0,
                  "the real disk producer rejected valid status");
            std::ofstream(boot / "transcript" / (std::to_string(agent) + ".jsonl"))
                << "{\"tick\":1,\"kind\":\"bound\",\"text\":\"\",\"request\":0,\"status\":\"limit\",\"turn\":1}\n"
                << "{\"tick\":2,\"kind\":\"done\",\"text\":\"\",\"request\":0,\"status\":\"prompt_refused\",\"turn\":2}\n";
        }
        for (unsigned bad = 0u; bad < 14u; ++bad) {
            Record record = policy_record(0u, count + 1u, 0u, 0u, 0u, 0u);
            switch (bad) {
            case 0u: record.header.type = AOTX_REC_CONSOLE; break;
            case 1u: record.header.cls = AOTX_CLASS_A; break;
            case 2u: record.header.writer = AOTX_WRITER_CONSOLE; break;
            case 3u: record.header.writer = AOTX_WRITER_AGENT_BASE; break;
            case 4u: record.header.flags = AOTX_FLAG_FRAGMENT; break;
            case 5u: record.header.body_len += 1u; break;
            case 6u: record.header.body_len -= 1u; break;
            case 7u: record.body.defaults = 1024u; break;
            case 8u: record.body.choices = 3u; break;
            case 9u: record.body.selected = 1u; break;
            case 10u: record.body.effective = 1u; break;
            case 11u: record.header.seq = 0u; break;
            case 12u: record.header.magic = 0u; break;
            default: record.body.agent = 256u; break;
            }
            check(aotx_tool_policy_stream_record(stream, &record.header) == 0,
                  "invalid status caused an output failure");
        }
        check(aotx_tool_policy_stream_lines(stream) == count
            && aotx_tool_policy_stream_refused(stream) == 14u, "invalid typed records reached the display stream");
        check(aotx_tool_policy_stream_sync(stream) == 0, "the policy stream did not sync");
        State state(root, root / "settings");
        check(state.open(), "the fixture replica did not open");
        state.tick(1.0);
        check(state.tool_policies().size() == count && state.agents().size() == count,
              "the replica omitted distinct conversation statuses");
        std::ofstream forged(boot / "console.log", std::ios::app);
        for (unsigned agent = 0u; agent < count; ++agent) {
            for (const char *prefix : {"", "conductor: ", "> say "})
                forged << prefix << "tools: agent " << agent
                       << " defaults 0 choices 0 selected 0 effective 0\n";
            forged << policy_line(agent, 0u, 0u, 0u, 0u) << '\n';
        }
        forged.close();
        state.tick(2.0);
        check(state.tool_policies().size() == count, "console text changed the number of policy rows");
        for (const auto &held : state.tool_policies())
            check(held.selected == (1u << (held.agent % AOTX_TOOL_POLICY_GROUPS))
                && held.sequence == held.agent + 1u, "generated console text replaced typed policy");
        for (const auto &agent : state.agents())
            check(!agent.reply_in_flight && !agent.reply_bound
                && agent.transcript.back().status == "prompt_refused",
                "input refusal retained a reply or continuation state");
        Record updated = policy_record(0u, count + 1u, 1023u, 0u, 1023u, 511u);
        check(aotx_tool_policy_stream_record(stream, &updated.header) == 0,
              "the new typed status was not written");
        aotx_tool_policy_stream_close(stream);
        std::ofstream(boot / "transcript/0.jsonl", std::ios::app)
            << "{\"tick\":3,\"kind\":\"reply\",\"text\":\"new reply\",\"request\":0,\"status\":\"stop\",\"turn\":3}\n";
        state.tick(3.0);
        check(state.tool_policies().front().defaults == 1023u,
              "a reported instance change did not update the view");
        check(state.agents().front().transcript.back().text == "new reply",
              "the conversation did not accept a reply after refusal");
        client::Client disconnected(root);
        tools::Query query;
        for (unsigned frame = 0u; frame < 300u; ++frame)
            tools::request_status(query, disconnected, "offline", "agent 0 tools", false);
        check(disconnected.take_results().empty() && !query.result.empty() && query.attempted,
              "offline frames queued repeated client errors");
        unsigned sent = 0u;
        for (unsigned frame = 0u; frame < 300u; ++frame)
            sent += tools::query_due(query, "offline", true, false) ? 1u : 0u;
        check(sent == 1u, "reconnect did not produce exactly one query");
        check(tools::query_due(query, "offline", true, true), "explicit refresh was not admitted");
        check(!tools::query_due(query, "offline", false, false)
            && tools::query_due(query, "offline", true, false), "another connection did not renew its query");
        check(tools::query_due(query, "new-boot", true, false), "a new boot did not renew its query");
        std::filesystem::remove_all(root);
    }
}
