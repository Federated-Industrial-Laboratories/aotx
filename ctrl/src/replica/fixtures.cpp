// Purpose: Check typed replica schemas and incremental state paths.
// Owns: Temporary fixture journals made during program startup.
// Launch shape: One interface thread runs one fixture sequence.
// Lifetime: Every fixture directory is removed before the check returns.
#include "replica/replica.hpp"

#include "replica/schema.hpp"
#include "replica/store.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <string>
#include <system_error>

namespace aotx::ctrl::replica {
bool verify_fixtures()
{
    Request request_fixture;
    std::string key;
    std::string value;
    std::string module_name;
    std::string module_kind;
    std::string fetch_name;
    std::string fetch_state;
    std::string loaded;
    std::string phase_word;
    PendingRequest pending_fixture;
    AgentState agent_fixture;
    TokenStat token_fixture;
    PageStat page_fixture;
    ModelParameters parameters_fixture;
    SteerVector steer_fixture;
    std::uint64_t fetched = 0u;
    std::uint64_t total = 0u;
    const std::string request_line =
        "{\"request\":41,\"agent\":2,\"turn\":3,\"tool\":\"fs_read\","
        "\"side\":\"host\",\"number\":3,\"arg\":\"path=hello.txt\","
        "\"deadline\":0,\"auth\":\"pending\",\"tick\":7}";
    const bool panel_fixtures =
        schema::request(request_line, request_fixture) &&
        request_fixture.authorization == "pending" &&
        !schema::request(request_line + "x", request_fixture) &&
        schema::setting_result("setting decode.reply_limit 37", key, value) &&
        key == "decode.reply_limit" && value == "37" &&
        !schema::setting_result("setting decode.reply_limit", key, value) &&
        schema::import_result("module check_skill skill import 4 from /tmp/check_skill",
                              module_name, module_kind) &&
        module_name == "check_skill" && module_kind == "skill" &&
        !schema::import_result("module check_skill other import 4 from /tmp/check_skill",
                               module_name, module_kind) &&
        schema::fetch_result("fetch language 7 of 19", fetch_name, fetched, total,
                             fetch_state) &&
        fetch_name == "language" && fetched == 7u && total == 19u &&
        !schema::fetch_result("fetch language 19 of 0", fetch_name, fetched, total,
                              fetch_state) &&
        schema::model_load("model language loaded model-q8.gguf at tick 37", "language",
                           loaded) && loaded == "model-q8.gguf" &&
        !schema::model_load("model language loaded model-q8.gguf at tick x", "language",
                            loaded) &&
        schema::pending_request("request 1042 pending fs_read agent 42 turn 2 path file-42.txt",
                                pending_fixture) &&
        pending_fixture.request == 1042u && pending_fixture.agent == 42u &&
        pending_fixture.turn == 2u && pending_fixture.tool == "fs_read" &&
        pending_fixture.path == "file-42.txt" &&
        !schema::pending_request("request x pending fs_read agent 42 turn 2 path file-42.txt",
                                 pending_fixture) &&
        schema::phase("placing 1", phase_word) && phase_word == "placing" &&
        schema::phase("replaying 2", phase_word) && phase_word == "replaying" &&
        schema::phase("running 3", phase_word) && phase_word == "running" &&
        schema::phase("closed 4", phase_word) && phase_word == "closed" &&
        !schema::phase("ready 5", phase_word) &&
        schema::agent_state("agent 7 turn role 2 parent 0 state 4 turn 3 ticks 91",
                            agent_fixture) &&
        agent_fixture.agent == 7u && agent_fixture.state == 4u && agent_fixture.turn == 3u &&
        !schema::agent_state("agent 7 turn role 2 parent 0 state 9 turn 3 ticks 91",
                             agent_fixture) &&
        schema::token_stat("{\"tick\":8,\"agent\":2,\"turn\":3,\"index\":4,"
                           "\"token\":17,\"logprob\":-0.25,\"entropy\":1.5,"
                           "\"think\":false}", token_fixture) &&
        token_fixture.index == 4u && !token_fixture.think &&
        !schema::token_stat("{\"tick\":8,\"agent\":2,\"turn\":3,\"index\":4,"
                            "\"token\":17,\"logprob\":0.25,\"entropy\":1.5,"
                            "\"think\":false}", token_fixture) &&
        schema::page_stat("{\"tick\":64,\"agent\":2,\"page\":9,"
                          "\"residency\":1,\"slots\":160,\"mass\":0.75}", page_fixture) &&
        page_fixture.page == 9u && page_fixture.residency == 1u &&
        !schema::page_stat("{\"tick\":64,\"agent\":2,\"page\":160,"
                           "\"residency\":1,\"slots\":160,\"mass\":0.75}", page_fixture) &&
        !schema::page_stat("{\"tick\":64,\"agent\":2,\"page\":9,"
                           "\"residency\":1,\"mass\":0.75}", page_fixture) &&
        !schema::page_stat("{\"tick\":64,\"agent\":2,\"page\":9,"
                           "\"residency\":1,\"slots\":4097,\"mass\":0.75}", page_fixture) &&
        schema::model_parameters("{\"name\":\"language\",\"parameters\":{"
            "\"temperature\":{\"default\":0.7,\"min\":0,\"max\":2}}}",
            parameters_fixture) && parameters_fixture.values.size() == 1u &&
        !schema::model_parameters("{\"name\":\"language\",\"parameters\":{"
            "\"temperature\":{\"default\":3,\"min\":0,\"max\":2}}}",
            parameters_fixture) &&
        schema::steer_vector("{\"name\":\"directness\",\"file\":"
                             "\"directness.aotxvec\",\"potency_nats\":0.9}",
                             steer_fixture) && steer_fixture.potency_nats == 0.9 &&
        !schema::steer_vector("{\"name\":\"directness\",\"file\":"
                              "\"directness.aotxvec\"}", steer_fixture);
    if (!panel_fixtures) return false;
    std::array<char, 40> pattern{};
    const std::string base = "/tmp/aotx_ctrl_replica_XXXXXX";
    std::copy(base.begin(), base.end(), pattern.begin());
    char *made = mkdtemp(pattern.data());
    if (made == nullptr) return false;
    const std::filesystem::path root(made);
    const std::filesystem::path boot = root / "0000000000000001";
    std::error_code error;
    std::filesystem::create_directories(boot / "transcript", error);
    std::filesystem::create_directories(root / "bus", error);
    std::filesystem::create_directories(root / "models/voice", error);
    if (error) return false;
    {
        std::ofstream(boot / "transcript/0.jsonl")
            << "{\"tick\":6,\"kind\":\"part\",\"text\":\"First \","
               "\"request\":0,\"status\":\"open\",\"turn\":1}\n"
            << "{\"tick\":7,\"kind\":\"part\",\"text\":\"second\","
               "\"request\":0,\"status\":\"open\",\"turn\":1}\n";
        std::ofstream(root / "bus/2000-01-01-aotx.jsonl")
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"system\",\"seq\":1,"
               "\"ts\":\"2000-01-01T00:00:00.000+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"sequence done slot 0\",\"tick\":8,"
               "\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n"
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"agent-0\",\"seq\":2,"
               "\"ts\":\"2000-01-01T00:00:00.100+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"request 41 pending fs_read agent 0 turn 1 path hello.txt\","
               "\"tick\":8,\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n"
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"agent-0\",\"seq\":3,"
               "\"ts\":\"2000-01-01T00:00:00.200+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"agent 0 spawned role 1 parent 0 state 1 turn 0 ticks 8\","
               "\"tick\":8,\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n";
        std::ofstream(root / "requests.jsonl")
            << "{\"request\":41,\"agent\":0,\"turn\":1,\"tool\":\"fs_read\","
               "\"side\":\"host\",\"number\":3,\"arg\":\"\\u001fpath=hello.txt\","
               "\"deadline\":500,\"auth\":\"pending\",\"tick\":7}\n";
        std::ofstream(root / "modules.jsonl")
            << "{\"name\":\"reader\",\"kind\":\"tool\",\"side\":\"host\","
               "\"dir\":\"tools/reader\",\"program\":\"run\",\"timeout\":30,"
               "\"authorize\":\"never\",\"import\":1,\"number\":17}\n";
        std::ofstream(root / "phase") << "running 1\n";
        std::ofstream(root / "settings") << "journal.dir = " << root.string() << "\n"
                                           << "models.dir = " << (root / "models").string()
                                           << "\n";
        std::ofstream(boot / "tokens.jsonl")
            << "{\"tick\":8,\"agent\":0,\"turn\":1,\"index\":0,\"token\":17,"
               "\"logprob\":-0.25,\"entropy\":1.5,\"think\":false}\n";
        std::ofstream(boot / "pages.jsonl")
            << "{\"tick\":64,\"agent\":0,\"page\":9,\"residency\":1,\"slots\":160,"
               "\"mass\":0.75}\n";
        std::ofstream(root / "models/parameters.jsonl")
            << "{\"name\":\"language\",\"parameters\":{\"temperature\":{"
               "\"default\":0.7,\"min\":0,\"max\":2}}}\n";
        std::ofstream(root / "models/steer.jsonl")
            << "{\"name\":\"directness\",\"file\":\"directness.aotxvec\","
               "\"potency_nats\":0.9}\n";
        std::ofstream(root / "models/voice/concise.profile")
            << "concise\n1.5\tbrief\n";
    }
    std::vector<ModelParameters> control_parameters;
    std::vector<SteerVector> control_vectors;
    std::vector<VoiceProfile> control_profiles;
    std::string control_reason;
    const bool catalog_valid = store::read_controls(root / "models", control_parameters,
        control_vectors, control_profiles, control_reason) && control_vectors.size() == 1u &&
        control_profiles.size() == 1u;
    std::ofstream(root / "models/voice/concise.profile", std::ios::app) << "bad line\n";
    const bool catalog_mutation = !store::read_controls(root / "models", control_parameters,
        control_vectors, control_profiles, control_reason);
    std::ofstream(root / "models/voice/concise.profile", std::ios::trunc)
        << "concise\n1.5\tbrief\n";
    State state(root, root / "settings");
    std::string configured;
    const bool opened = state.open();
    const bool folded = opened && state.agents().size() == 1u &&
                        state.agents()[0].transcript.size() == 1u &&
                        state.agents()[0].transcript[0].kind == "part" &&
                        state.agents()[0].transcript[0].text == "First second" &&
                        state.agents()[0].part_lines == 2u &&
                        state.pending_requests().size() == 1u &&
                        state.agent_states().size() == 1u &&
                        state.take_results().size() == 1u;
    {
        std::ofstream transcript(boot / "transcript/0.jsonl", std::ios::app);
        transcript << "{\"tick\":8,\"kind\":\"reply\",\"text\":\"First second\","
                      "\"request\":0,\"status\":\"\",\"turn\":1}\n"
                   << "{\"tick\":9,\"kind\":\"bound\",\"text\":\"\","
                      "\"request\":0,\"status\":\"limit\",\"turn\":1}\n"
                   << "{\"tick\":10,\"kind\":\"grant\",\"tool\":\"fs_read\","
                      "\"request\":41,\"status\":\"granted\",\"turn\":1}\n"
                   << "{\"tick\":11,\"kind\":\"unknown\"}\n";
    }
    {
        std::ofstream(root / "bus/2000-01-01-aotx.jsonl", std::ios::app)
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"system\",\"seq\":2,"
               "\"ts\":\"2000-01-01T00:00:01.000+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"model language loaded model-q8.gguf at tick 37\","
               "\"tick\":9,\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n";
    }
    state.tick(1.0);
    const std::vector<std::string> final_results = state.take_results();
    const bool valid = catalog_valid && catalog_mutation && folded &&
                       setting_value(root / "settings", "journal.dir", configured) &&
                       configured == root.string() && state.phase() == "running" &&
                       state.agents().size() == 1u &&
                       state.agents()[0].transcript.size() == 3u &&
                       state.agents()[0].transcript[0].kind == "reply" &&
                       state.agents()[0].transcript[0].text == "First second" &&
                       state.agents()[0].transcript[1].kind == "bound" &&
                       state.agents()[0].transcript[1].status == "limit" &&
                       state.agents()[0].transcript[2].kind == "grant" &&
                       state.agents()[0].fold_replaced && state.agents()[0].reply_bound &&
                       state.language_model() == "model-q8.gguf" &&
                       state.notes().size() == 4u && state.requests().size() == 1u &&
                       state.requests()[0].argument == "\x1fpath=hello.txt" &&
                       state.pending_requests().empty() &&
                       state.modules().size() == 1u && final_results.size() == 2u &&
                       state.tokens().size() == 1u && state.pages().size() == 1u &&
                       state.model_parameters().size() == 1u &&
                       state.steer_vectors().size() == 1u &&
                       state.voice_profiles().size() == 1u &&
                       std::find(final_results.begin(), final_results.end(),
                                 "model language loaded model-q8.gguf at tick 37") !=
                           final_results.end();
    std::filesystem::remove_all(root, error);
    return valid;
}

} // namespace aotx::ctrl::replica
