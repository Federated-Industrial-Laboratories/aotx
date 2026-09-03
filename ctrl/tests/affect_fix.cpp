// Purpose: Check affect schemas, calibration data, and bounded rings.
// Owns: Temporary affect, quality, calibration, and probe files.
// Launch shape: One host thread applies each bounded fixture in order.
// Lifetime: Every temporary file is removed before the check returns.
#include "affect_fix.hpp"

#include "replica/schema.hpp"
#include "replica/stats.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

void check(bool value, const char *text, int &applied, int &failed)
{
    ++applied;
    if (value) return;
    ++failed;
    std::printf("ctrl fix: %s\n", text);
}

std::filesystem::path temp_root()
{
    std::array<char, 48> pattern{};
    const std::string base = "/tmp/aotx_ctrl_affect_fix_XXXXXX";
    std::copy(base.begin(), base.end(), pattern.begin());
    char *made = mkdtemp(pattern.data());
    return made == nullptr ? std::filesystem::path{} : std::filesystem::path(made);
}

std::string affect_line(unsigned turn, unsigned agent = 0u)
{
    return "{\"tick\":" + std::to_string(turn) + ",\"agent\":" +
           std::to_string(agent) + ",\"turn\":" + std::to_string(turn) +
           ",\"kind\":\"trace\",\"prompt\":[0.4,-0.1,0,0],"
           "\"reply\":[0.6,0.2,0,0],\"guard\":[0.1,-0.3],"
           "\"logprob\":-0.82,\"entropy\":1.4,\"rows\":57,\"think\":0,"
           "\"reason\":[\"stop\",\"tool_ok\"],\"effective\":[0.31,0.05,0,0],"
           "\"flags\":1}";
}

std::string quality_line(unsigned turn, unsigned agent = 0u)
{
    return "{\"tick\":" + std::to_string(turn) + ",\"agent\":" +
           std::to_string(agent) + ",\"turn\":" + std::to_string(turn) +
           ",\"coherence_prompt\":null,\"coherence_turn\":null,"
           "\"repetition\":0.06,\"tokens\":212,\"limit\":256,"
           "\"limit_hit\":0,\"refusal\":0,\"guard\":[0.1,-0.3],\"flags\":0}";
}

std::string calibration_line()
{
    return "{\"role\":\"language\",\"axes\":[\"valence\",\"arousal\"],"
           "\"rows\":[\"valence\",\"arousal\",\"sycophancy\",\"refusal\"],"
           "\"M\":[[1.8,0.09],[-0.04,0.37],[0.28,0.05],[-0.08,0.03]],"
           "\"K\":[[0.08,0.01],[0.01,0.09]],\"ratio\":[4.08,3.97],"
           "\"dominant\":1,\"orthogonal\":1}";
}

std::string probe_line()
{
    return "{\"name\":\"valence\",\"file\":\"affect/valence.aotxprb\","
           "\"axis\":0,\"layer\":24,\"accuracy\":0.9375}";
}

void schema_cases(int &applied, int &failed)
{
    using namespace aotx::ctrl::replica;
    AffectTrace trace;
    QualityLine quality;
    Calibration calibration;
    ProbeAccuracy probe;
    check(schema::affect_trace(affect_line(3u), trace) && trace.trace && trace.turn == 3u &&
              trace.reason.size() == 2u && trace.effective[0] == 0.31,
          "the affect trace line did not parse", applied, failed);
    check(schema::quality_line(quality_line(3u), quality) &&
              !quality.coherence_prompt.has_value() && !quality.coherence_turn.has_value() &&
              quality.repetition == 0.06,
          "the quality line did not keep null coherence figures", applied, failed);
    std::string bad_affect = affect_line(3u);
    bad_affect.replace(bad_affect.find("[0.31,0.05,0,0]"), 17u, "[0.31,0.05,0]");
    check(!schema::affect_trace(bad_affect, trace),
          "a malformed affect trace line was accepted", applied, failed);
    std::string bad_quality = quality_line(3u);
    bad_quality.replace(bad_quality.find("\"flags\":0"), 9u, "\"flags\":1");
    check(!schema::quality_line(bad_quality, quality),
          "a malformed quality line was accepted", applied, failed);
    check(schema::calibration(calibration_line(), calibration) &&
              calibration.K[1][1] == 0.09 && calibration.ratio[0] == 4.08 &&
              calibration.rows[2] == "sycophancy" && calibration.response[3][0] == -0.08,
          "the calibration line did not keep its instrument figures", applied, failed);
    std::string bad_calibration = calibration_line();
    bad_calibration.replace(bad_calibration.find("[0.01,0.09]"), 11u, "[0.02,0.09]");
    check(!schema::calibration(bad_calibration, calibration),
          "a malformed calibration line was accepted", applied, failed);
    check(schema::probe_accuracy(probe_line(), probe) && probe.name == "valence" &&
              probe.axis == 0u && probe.accuracy == 0.9375,
          "the probe accuracy line did not parse", applied, failed);
    check(!schema::probe_accuracy(probe_line() + "x", probe),
          "a malformed probe accuracy line was accepted", applied, failed);
}

void ring_cases(int &applied, int &failed)
{
    using namespace aotx::ctrl::replica;
    const std::filesystem::path root = temp_root();
    const std::filesystem::path boot = root / "0000000000000001";
    const std::filesystem::path store = root / "models";
    std::filesystem::create_directories(boot);
    std::filesystem::create_directories(store / "affect");
    std::ofstream affect(boot / "affect.jsonl");
    std::ofstream quality(boot / "quality.jsonl");
    for (unsigned turn = 1u; turn <= 33u; ++turn) {
        affect << affect_line(turn) << '\n';
        quality << quality_line(turn) << '\n';
    }
    affect << "{\"tick\":34,\"agent\":0,\"turn\":34,\"kind\":\"state\"}\n"
           << "{\"kind\":\"trace\"}\n";
    quality << "{\"tick\":34}\n";
    affect.close();
    quality.close();
    std::ofstream(store / "affect/calibration.jsonl") << calibration_line() << '\n';
    std::ofstream(store / "probes.jsonl") << probe_line() << '\n';
    stats::Reader reader;
    std::vector<std::string> results;
    reader.read(boot, store, 1.0, results);
    check(reader.affect_traces().size() == 32u &&
              reader.affect_traces().front().turn == 2u &&
              reader.affect_traces().back().turn == 33u &&
              reader.quality_lines().size() == 32u &&
              reader.quality_lines().front().turn == 2u &&
              reader.quality_lines().back().turn == 33u && results.size() == 2u,
          "the measurement rings did not keep the last 32 turns", applied, failed);
    check(reader.calibration().has_value() && reader.probe_accuracies().size() == 1u,
          "the store figures did not enter the replica", applied, failed);
    std::ofstream(store / "affect/calibration.jsonl", std::ios::app) << "{}\n";
    reader.read(boot, store, 2.0, results);
    check(!reader.calibration().has_value() && results.size() == 3u &&
              results.back() == "The calibration line 2 was refused.",
          "a malformed calibration line was not counted and stated", applied, failed);
    std::filesystem::remove_all(root);
}

} // namespace

void aotx_ctrl_affect_fix(int &applied, int &failed)
{
    schema_cases(applied, failed);
    ring_cases(applied, failed);
}
