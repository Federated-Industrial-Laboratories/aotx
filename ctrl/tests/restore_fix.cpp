// Purpose: Check fresh and restored launch arguments without device work.
// Owns: Isolated settings, journals, and argument-reporting child processes.
// Threading: One host thread runs batches of 1 and 64 distinct instances.
// Lifetime: Every child ends before its instance files are removed.
#include "restore_fix.hpp"
#include "instances/lifecycle.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <vector>

namespace {
void check(bool value, const char *reason, int &applied, int &failed)
{
    ++applied;
    if (!value) { ++failed; std::printf("ctrl restore: %s\n", reason); }
}

std::string read(const std::filesystem::path &path)
{
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void batch(unsigned count, int &applied, int &failed)
{
    char pattern[] = "/tmp/aotx_ctrl_restore_XXXXXX";
    char *made = mkdtemp(pattern);
    check(made != nullptr, "the fixture directory did not open", applied, failed);
    if (!made) return;
    const std::filesystem::path root(made);
    std::filesystem::create_directory(root / "build");
    std::filesystem::create_symlink("/bin/echo", root / "build/aotx_boot");
    aotx::ctrl::instances::Lifecycle lifecycle;
    std::vector<aotx::ctrl::instances::Definition> definitions;
    std::vector<std::string> settings;
    for (unsigned index = 0; index < count; ++index) {
        aotx::ctrl::instances::Definition definition;
        const auto input = root / std::to_string(index);
        definition.name = "Fixture " + std::to_string(index);
        definition.journal = input / "journal";
        definition.settings = input / "settings";
        definition.build = root / "build";
        definition.models = input / "models";
        const bool created = lifecycle.create(definition);
        check(created, "the instance did not create", applied, failed);
        if (!created) { std::filesystem::remove_all(root); return; }
        definitions.push_back(definition);
        settings.push_back(read(definition.settings));
    }
    check(lifecycle.instances().size() == count, "the batch omitted an instance", applied, failed);
    for (bool restore : {false, true}) {
        for (unsigned step = 0; step < count; ++step) {
            const unsigned index = count - step - 1;
            const auto &definition = definitions[index];
            check(lifecycle.start(index, restore), "the instance did not start", applied, failed);
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < until) {
                lifecycle.tick(0);
                if (lifecycle.instances()[index].process < 0) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto instance = lifecycle.instances()[index];
            check(instance.process < 0 && instance.result.find("status 0:") != std::string::npos,
                  "the argument child did not exit successfully", applied, failed);
            check(instance.result.find("--settings " + definition.settings.string()) != std::string::npos,
                  "the launch used another instance's settings", applied, failed);
            check((instance.result.find("--restore") != std::string::npos) == restore,
                  "the launch did not retain its fresh or restore selection", applied, failed);
            check(read(definition.settings) == settings[index],
                  "the launch changed the saved settings", applied, failed);
        }
    }
    std::filesystem::remove_all(root);
}
} // namespace

void aotx_ctrl_restore_fix(int &applied, int &failed)
{
    batch(1, applied, failed);
    batch(64, applied, failed);
}
