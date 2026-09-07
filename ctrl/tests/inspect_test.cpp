// Purpose: Check bounded model inspection with distinct local child reports.
// Owns: Temporary executables, source files, and N=1 and N=64 assertions.
// Threading: One test caller polls asynchronous inspection children.
// Lifetime: Each test scope removes its temporary files.
#include "model/inspect.hpp"
#include "inspect_fixture.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sys/stat.h>
#include <thread>
#include <vector>

namespace {

namespace model = aotx::ctrl::model;
namespace fixture = aotx::ctrl::test;
namespace fs = std::filesystem;
using namespace std::chrono_literals;
unsigned checks = 0, bad = 0;

void check(bool condition, const std::string &message)
{
    ++checks;
    if (!condition) { ++bad; std::fprintf(stderr, "FAIL %s\n", message.c_str()); }
}

struct Files {
    fs::path root, build;
    Files()
    {
        char pattern[] = "/tmp/aotx-inspect-XXXXXX";
        char *path = ::mkdtemp(pattern);
        if (!path) std::abort();
        root = path;
        build = root / "build with space $literal";
        fs::create_directory(build);
        std::array<char, 8192> exe{};
        const ssize_t length = ::readlink("/proc/self/exe", exe.data(), exe.size());
        if (length <= 0 || static_cast<std::size_t>(length) >= exe.size()) std::abort();
        fs::create_symlink(std::string(exe.data(), static_cast<std::size_t>(length)),
                           build / "aotx_models");
    }
    ~Files() { fs::remove_all(root); }
    std::string source(const std::string &mode, unsigned id)
    {
        const fs::path file = root / ("file " + mode + " " + std::to_string(id) + " '$(false);\\.gguf");
        std::ofstream out(file);
        out << mode << ' ' << id << '\n';
        out.close();
        if (!out) std::abort();
        return file.string();
    }
};

void finish(model::InspectAction &action)
{
    const auto end = std::chrono::steady_clock::now() + 5s;
    while (action.running() && std::chrono::steady_clock::now() < end) {
        action.tick();
        std::this_thread::sleep_for(1ms);
    }
    check(!action.running(), "the action finishes within the test deadline");
    if (action.running()) action.cancel();
}

void facts(const model::InspectAction &action, unsigned id, const std::string &source,
           const fs::path &build, bool supported, bool remote = false)
{
    check(action.status() == (supported ? model::InspectStatus::complete : model::InspectStatus::unsupported),
          "the completed header support status matches the child report");
    check(action.input() == source && action.build() == build, "the request retains its exact source and build");
    check(action.header().has_value(), "a complete report has structured header facts");
    if (!action.header()) return;
    const auto &h = *action.header();
    check(h.architecture == "family_" + std::to_string(id), "the architecture belongs to this source");
    check(h.pre_tokenizer == "tokenizer_" + std::to_string(id), "the tokenizer belongs to this source");
    check(h.tensors == id + 100 && h.layers == id + 1 && h.hidden == id + 128 &&
          h.vocabulary == id + 1000, "the complete dimensions retain their distinct values");
    check(h.file_bytes == id + 100000 && h.header_bytes == id + 10000 &&
          h.chat_template_bytes == id + 200, "the byte counts retain their distinct values");
    check(h.chat_template_sha256 == std::string(64, "0123456789abcdef"[id % 16]),
          "the template digest belongs to this source");
    check(h.block_types.size() == 1 && h.block_types[0].id == id + 10 &&
          h.block_types[0].count == id + 100 && h.block_types[0].name == "Q8_0" &&
          h.block_types[0].supported, "the block type facts retain their identity and count");
    check(h.layer_types.size() == 1 && h.layer_types[0].name == "attention" &&
          h.layer_types[0].count == id + 1, "the layer type facts retain their count");
    check(h.build_support == supported && h.layer_sets_supported == supported &&
          h.pre_tokenizer_supported && h.tokenizer_model_supported && h.rotary_pairs_supported,
          "the support fields retain the report values");
    check(h.received_bytes.has_value() == remote && (!remote || *h.received_bytes == id + 20000),
          "only remote inspection reports received bytes");
}

void batches(Files &files, unsigned count)
{
    const unsigned before = checks;
    std::vector<std::unique_ptr<model::InspectAction>> actions;
    std::vector<std::string> sources;
    for (unsigned i = 0; i < count; ++i) {
        actions.push_back(std::make_unique<model::InspectAction>());
        sources.push_back(files.source("good", i));
        check(actions.back()->start(files.build, sources.back()), "start each distinct child without a shell");
    }
    for (unsigned i = 0; i < count; ++i) {
        finish(*actions[i]);
        facts(*actions[i], i, sources[i], files.build, true);
    }
    const std::vector<std::string> malformed = {
        "empty", "partial", "missing_support", "duplicate_support", "run_yes", "missing_run",
        "wrong_file", "bad_support", "overflow", "negative", "number_tail", "bad_digest",
        "duplicate_block", "duplicate_layer", "missing_block", "block_overflow", "layer_overflow",
        "missing_layer", "short_layers",
        "bad_escape", "control", "extra_field", "received_local", "stderr", "signal", "exit2", "exit1"
    };
    for (unsigned i = 0; i < count; ++i) {
        auto &action = *actions[i];
        for (const auto &mode : malformed) {
            check(action.start(files.build, files.source(mode, i)), "start the " + mode + " report");
            check(!action.header(), "replacement removes the preceding report");
            finish(action);
            check(action.status() == model::InspectStatus::failed && !action.header(),
                  mode + " cannot become supported or unsupported");
        }
        for (const auto &mode : {"unsupported0", "unsupported1", "late_tail", "good"}) {
            const std::string source = files.source(mode, i);
            check(action.start(files.build, source), "start a repeated complete report");
            finish(action);
            facts(action, i, source, files.build, std::string(mode).find("unsupported") != 0);
        }
        check(action.start(files.build, files.source("unsupported_layers", i)),
              "start an unsupported report with unmatched layers");
        finish(action);
        check(action.status() == model::InspectStatus::unsupported && action.header() &&
                  !action.header()->layer_sets_supported && action.header()->layer_types.empty() &&
                  action.header()->layers == i + 1,
              "unsupported unmatched layers remain visible as unsupported facts");
        const std::string remote = "https://invalid.example/case-" + std::to_string(i) + "?a='$(false)'";
        check(action.start(files.build, remote), "pass the URL unchanged to the local test executable");
        finish(action);
        facts(action, i, remote, files.build, true, true);
    }
    std::printf("inspection N=%u: %u checks\n", count, checks - before);
}

void lifecycle(Files &files, unsigned count)
{
    const unsigned before = checks;
    for (unsigned i = 0; i < count; ++i) {
        model::InspectAction timed({131072, 100ms});
        check(timed.start(files.build, files.source("hang", i)), "start a child that does not complete");
        finish(timed);
        check(timed.status() == model::InspectStatus::failed && !timed.header() &&
              timed.message().find("time limit") != std::string::npos, "the deadline refuses a stalled child");
        check(timed.start(files.build, files.source("held_pipe", i)), "start a report with an open child pipe");
        finish(timed);
        check(timed.status() == model::InspectStatus::failed && !timed.header() &&
              timed.output().find("run_verified=no\n") != std::string::npos,
              "an exited child cannot pass while another writer retains the pipe");
        model::InspectAction action;
        check(action.start(files.build, files.source("hang", i)), "start the action to cancel");
        action.cancel();
        check(action.status() == model::InspectStatus::cancelled && !action.header() &&
              action.output().empty(), "cancellation removes all previous result bytes and facts");
        const std::string source = files.source("good", i);
        check(action.start(files.build, source), "start immediately after cancellation");
        finish(action);
        facts(action, i, source, files.build, true);
        check(action.start(files.build, files.source("late_tail", i)), "start an old selection");
        action.tick();
        const std::string changed = files.source("good", i + 100);
        check(action.start(files.build, changed), "replace the selection while its child runs");
        check(!action.header() && action.output().empty(), "replacement clears old data before another poll");
        finish(action);
        facts(action, i + 100, changed, files.build, true);
        action.cancel();
        action.tick();
        check(!action.header(), "a completed result cannot return after cancellation");
    }
    std::printf("lifecycle N=%u: %u checks\n", count, checks - before);
}

void refusals(Files &files)
{
    model::InspectAction action;
    const auto source = files.source("good", 3);
    check(action.status() == model::InspectStatus::idle && !action.header(), "a new action is idle without facts");
    check(!action.start(files.root / "missing", source), "a missing executable is refused");
    check(action.status() == model::InspectStatus::failed && !action.header(), "a missing child has no facts");
    const fs::path other = files.root / "other-build";
    fs::create_directory(other);
    std::ofstream(other / "aotx_models") << "no executable header\n";
    check(!action.start(other, source), "a file without execute access is refused");
    ::chmod((other / "aotx_models").c_str(), 0700);
    check(!action.start(other, source), "a malformed executable is refused without a shell fallback");
    check(!action.start(files.build, ""), "an empty source is refused");
    check(!action.start(files.build, std::string("path\0tail", 9)), "a source with a zero byte is refused");
    check(action.start(files.build, source), "a valid request can follow a refused request");
    finish(action);
    facts(action, 3, source, files.build, true);
    check(!action.start(other, source) && action.build() == other && !action.header(),
          "a changed invalid build cannot retain a previous successful result");
    model::InspectAction small({256, 1s});
    check(small.start(files.build, source), "start an action with a small output bound");
    finish(small);
    check(small.status() == model::InspectStatus::failed && small.output().size() <= 256 && !small.header(),
          "output beyond the configured limit is refused and remains bounded");
    check(action.start(files.build, files.source("oversize", 1)), "start an output flood");
    finish(action);
    check(action.status() == model::InspectStatus::failed && action.output().size() <= 131072 && !action.header(),
          "a valid prefix cannot pass when later output exceeds the byte limit");
    check(action.start(files.build, files.source("hang", 1)), "start a child for destructor cleanup");
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 3 && std::string(argv[1]) == "inspect") return fixture::inspect_child(argv[2]);
    if (argc != 1) return 2;
    Files files;
    for (unsigned count : {1u, 64u}) { batches(files, count); lifecycle(files, count); }
    refusals(files);
    std::printf("inspection: %u checks, %u failures\n", checks, bad);
    return bad == 0 ? 0 : 1;
}
