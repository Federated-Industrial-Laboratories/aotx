// Purpose: Drive one complete live instance lifecycle and mirror exchange.
// Owns: One temporary instance manager and one typed telemetry reader.
// Launch shape: One process polls one headless child and its replica surfaces.
// Lifetime: The manager always stops an owned boot before this process ends.
#include "instances/lifecycle.hpp"
#include "chat/actions.hpp"
#include "monitor/telemetry.hpp"
#include "replica/replica.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::fputs("usage: aotx_ctrl_live_client <boot> <models> <run-directory>\n", stderr);
        return 2;
    }
    const std::filesystem::path boot = std::filesystem::canonical(argv[1]);
    const std::filesystem::path models = std::filesystem::canonical(argv[2]);
    const std::filesystem::path run = std::filesystem::absolute(argv[3]);
    aotx::ctrl::instances::Lifecycle lifecycle;
    aotx::ctrl::monitor::Telemetry telemetry;
    aotx::ctrl::instances::Definition definition;
    definition.name = "Live check instance";
    definition.journal = run / "journal";
    definition.settings = run / "instance.settings";
    definition.build = boot.parent_path();
    definition.models = models;
    if (!lifecycle.create(definition)) {
        std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
        return 1;
    }
    std::puts("live client: the lifecycle created the instance and its settings");
    if (!lifecycle.start(0u)) {
        std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
        return 1;
    }
    std::puts("live client: the lifecycle started aotx_boot headless");

    const auto start = std::chrono::steady_clock::now();
    std::uint64_t first_tick = 0u;
    bool running = false;
    bool said = false;
    bool advanced = false;
    bool stop_sent = false;
    bool reply_stopped = false;
    bool stopping = false;
    while (std::chrono::steady_clock::now() - start < std::chrono::minutes(3)) {
        const double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        lifecycle.tick(now);
        telemetry.tick(lifecycle.mirror_descriptor(0u), now);
        for (const std::string &result : lifecycle.take_results()) {
            std::printf("live client: lifecycle result: %s\n", result.c_str());
        }
        const std::vector<aotx::ctrl::instances::LiveInstance> items = lifecycle.instances();
        if (items.empty()) return 1;
        const auto &instance = items.front();
        if (!running && instance.state == aotx::ctrl::instances::LiveState::running &&
            instance.phase == "running" && instance.connection == "connected") {
            running = true;
            std::puts("live client: the running phase and socket connection are ready");
        }
        const auto &sample = telemetry.mirror();
        if (running && !said && sample.available) {
            first_tick = sample.tick;
            std::printf("live client: mirror sample one tick %llu sequence %llu\n",
                        static_cast<unsigned long long>(sample.tick),
                        static_cast<unsigned long long>(sample.sequence));
            if (!lifecycle.send(0u, "say Write the integers from one through five hundred.")) {
                std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
                return 1;
            }
            said = true;
            std::puts("live client: sent the long reply request");
        }
        aotx::ctrl::replica::State *replica = lifecycle.replica(0u);
        if (said && !stop_sent && replica != nullptr && !replica->agents().empty() &&
            replica->agents()[0].reply_in_flight) {
            const std::string command = aotx::ctrl::chat::stop_command(
                replica->agents()[0].id);
            if (!lifecycle.send(0u, command)) {
                std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
                return 1;
            }
            stop_sent = true;
            std::printf("live client: chat sent %s mid-reply\n", command.c_str());
        }
        if (stop_sent && !reply_stopped && replica != nullptr && !replica->agents().empty()) {
            for (const auto &event : replica->agents()[0].transcript) {
                if (event.kind == "done" && event.status == "stopped") {
                    reply_stopped = true;
                    std::puts("live client: the done record states stopped");
                    break;
                }
            }
        }
        if (said && !advanced && sample.available && sample.tick > first_tick) {
            advanced = true;
            std::printf("live client: mirror sample two tick %llu sequence %llu rate %.1f Hz\n",
                        static_cast<unsigned long long>(sample.tick),
                        static_cast<unsigned long long>(sample.sequence), sample.tick_rate);
        }
        if (advanced && reply_stopped && !stopping) {
            if (!lifecycle.stop(0u)) {
                std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
                return 1;
            }
            stopping = true;
            std::puts("live client: sent SIGTERM through the lifecycle");
        }
        if (stopping && instance.state == aotx::ctrl::instances::LiveState::stopped &&
            instance.phase == "closed" && instance.process < 0) {
            std::puts("live client: the lifecycle confirmed the closed phase");
            std::printf("live client: NVML result: %s\n", telemetry.card_result().c_str());
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::fprintf(stderr,
                 "live client: timeout running=%d said=%d stop=%d done=%d advanced=%d "
                 "stopping=%d\n",
                 running ? 1 : 0, said ? 1 : 0, stop_sent ? 1 : 0,
                 reply_stopped ? 1 : 0, advanced ? 1 : 0, stopping ? 1 : 0);
    return 1;
}
