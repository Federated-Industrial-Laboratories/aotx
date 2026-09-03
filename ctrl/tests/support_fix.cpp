// Purpose: Check persona storage and bounded child termination.
// Owns: Temporary persona files, pipes, and child processes.
// Launch shape: One host thread applies each bounded fixture in order.
// Lifetime: Every temporary resource ends before the check returns.
#include "support_fix.hpp"

#include "chat/persona.hpp"
#include "instances/lifecycle.hpp"
#include "process/child.hpp"

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

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
    const std::string base = "/tmp/aotx_ctrl_support_fix_XXXXXX";
    std::copy(base.begin(), base.end(), pattern.begin());
    char *made = mkdtemp(pattern.data());
    return made == nullptr ? std::filesystem::path{} : std::filesystem::path(made);
}

void persona_storage_case(int &applied, int &failed)
{
    const std::filesystem::path root = temp_root();
    aotx::ctrl::chat::persona::Store store(root / "personas");
    std::string result;
    std::string voice;
    const std::filesystem::path journal = root / "journal";
    check(store.save_default(journal, "Default voice.", result) &&
              store.default_voice(journal) == "Default voice.",
          "the instance persona did not persist", applied, failed);
    check(store.save_override(journal, 3u, "Other voice.", result) &&
              store.override_voice(journal, 3u, voice) && voice == "Other voice.",
          "the conversation persona did not persist", applied, failed);
    check(store.save_override(journal, 3u, "", result) &&
              !store.override_voice(journal, 3u, voice),
          "the conversation persona did not return to the instance default", applied, failed);
    std::filesystem::remove_all(root);
}

void child_escalation_case(int &applied, int &failed)
{
    int ready[2];
    check(pipe(ready) == 0, "the child fixture pipe did not open", applied, failed);
    const pid_t child = fork();
    if (child == 0) {
        signal(SIGTERM, SIG_IGN);
        const ssize_t notified = write(ready[1], "r", 1);
        (void)notified;
        for (;;) pause();
    }
    char byte = '\0';
    check(read(ready[0], &byte, 1) == 1, "the child fixture did not start", applied, failed);
    close(ready[0]);
    close(ready[1]);
    const auto start = std::chrono::steady_clock::now();
    const aotx::ctrl::process::End ended = aotx::ctrl::process::end_child(child);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    check(ended == aotx::ctrl::process::End::kill && seconds < 1.5,
          "a SIGTERM-ignoring child did not receive bounded SIGKILL escalation",
          applied, failed);
}

void child_last_line_case(int &applied, int &failed)
{
    const std::filesystem::path root = temp_root();
    std::filesystem::create_directories(root / "build");
    std::filesystem::create_directories(root / "models");
    std::filesystem::create_symlink("/bin/echo", root / "build/aotx_boot");
    aotx::ctrl::instances::Definition definition;
    definition.name = "Line fixture";
    definition.journal = root / "journal";
    definition.settings = root / "settings";
    definition.build = root / "build";
    definition.models = root / "models";
    aotx::ctrl::instances::Lifecycle lifecycle;
    check(lifecycle.create(definition) && lifecycle.start(0u),
          "the child line fixture did not start", applied, failed);
    for (unsigned index = 0u; index < 100u; ++index) {
        lifecycle.tick(static_cast<double>(index) / 100.0);
        if (lifecycle.instances()[0].process < 0) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const std::string result = lifecycle.instances()[0].result;
    check(result.find("child died with status 0: --settings") != std::string::npos,
          "the child result omitted its piped last line", applied, failed);
    std::filesystem::remove_all(root);
}

} // namespace

void aotx_ctrl_support_fix(int &applied, int &failed)
{
    persona_storage_case(applied, failed);
    child_escalation_case(applied, failed);
    child_last_line_case(applied, failed);
}
