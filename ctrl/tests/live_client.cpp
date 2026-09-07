// Purpose: Drive one complete live instance lifecycle and mirror exchange.
// Owns: One temporary instance manager and one typed telemetry reader.
// Launch shape: One process polls one headless child and its replica surfaces.
// Lifetime: The manager always stops an owned boot before this process ends.
#include "instances/lifecycle.hpp"
#include "chat/actions.hpp"
#include "chat/persona.hpp"
#include "model/controls.hpp"
#include "monitor/telemetry.hpp"
#include "replica/replica.hpp"

#include <chrono>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

int main(int argc, char **argv)
{
    unsigned card = 0;
    bool valid = argc == 4 || argc == 6;
    if (argc == 6) {
        const std::string value = argv[5];
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), card);
        valid = std::string(argv[4]) == "--card" && parsed.ec == std::errc{} &&
                parsed.ptr == value.data() + value.size();
    }
    if (!valid) {
        std::fputs("usage: aotx_ctrl_live_client <boot> <models> <run-directory> "
                   "[--card <index>]\n", stderr);
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
    definition.card = card;
    if (!lifecycle.create(definition)) {
        std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
        return 1;
    }
    std::puts("live client: the lifecycle created the instance and its settings");
    std::printf("live client: selected card %u\n", card);
    const std::filesystem::path preset_path = run / "balanced.preset";
    std::ofstream(preset_path) << "decode.temperature = 0.8\n"
                                 "decode.top_p = 0.95\n";
    aotx::ctrl::model::Preset preset;
    std::string reason;
    if (!aotx::ctrl::model::read_preset(preset_path, preset, reason)) {
        std::fprintf(stderr, "live client: %s\n", reason.c_str());
        return 1;
    }
    aotx::ctrl::chat::persona::RoleModule persona;
    if (!aotx::ctrl::chat::persona::write_role_module(
            run / "persona-source", "Use calm and direct sentences.", persona, reason)) {
        std::fprintf(stderr, "live client: %s\n", reason.c_str());
        return 1;
    }
    std::printf("live client: composed persona role %s in %s\n", persona.name.c_str(),
                persona.directory.c_str());
    if (!lifecycle.start(0u)) {
        std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
        return 1;
    }
    std::puts("live client: the lifecycle started aotx_boot headless");

    const auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::minutes(6);
    std::uint64_t first_tick = 0u;
    bool running = false;
    bool controls_sent = false;
    bool preset_confirmed = false;
    bool persona_installed = false;
    bool spawn_sent = false;
    bool persona_spawned = false;
    bool persona_said = false;
    bool persona_replied = false;
    bool tokens_joined = false;
    bool said = false;
    bool advanced = false;
    bool stop_sent = false;
    bool reply_stopped = false;
    bool stopping = false;
    while (std::chrono::steady_clock::now() < deadline) {
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
            deadline = std::chrono::steady_clock::now() + std::chrono::minutes(3);
            std::puts("live client: the running phase and socket connection are ready");
        }
        const auto &sample = telemetry.mirror();
        aotx::ctrl::replica::State *replica = lifecycle.replica(0u);
        if (running && !controls_sent && sample.available && replica != nullptr &&
            !replica->agents().empty()) {
            first_tick = sample.tick;
            std::printf("live client: mirror sample one tick %llu sequence %llu\n",
                        static_cast<unsigned long long>(sample.tick),
                        static_cast<unsigned long long>(sample.sequence));
            const unsigned agent = replica->agents()[0].id;
            for (const std::string &command :
                 aotx::ctrl::model::preset_commands(agent, preset)) {
                if (!lifecycle.send(0u, command)) {
                    std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
                    return 1;
                }
                std::printf("live client: preset sent %s\n", command.c_str());
            }
            if (!lifecycle.send(0u, "import " + persona.directory.string())) {
                std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
                return 1;
            }
            controls_sent = true;
            std::printf("live client: import sent for role %s\n", persona.name.c_str());
        }
        if (controls_sent && replica != nullptr) {
            unsigned acknowledgments = 0u;
            for (const auto &value : preset.values) {
                const std::string wanted = "agent: " + value.first + " " + value.second +
                                           " changes at the next turn";
                for (const auto &note : replica->notes()) {
                    if (note.text == wanted) { ++acknowledgments; break; }
                }
            }
            if (!preset_confirmed && acknowledgments == preset.values.size()) {
                preset_confirmed = true;
                std::printf("live client: preset %s has %u acknowledgments\n",
                            preset.name.c_str(), acknowledgments);
            }
            for (const auto &note : replica->notes()) {
                if (note.text == "import: the role " + persona.name + " is installed") {
                    persona_installed = true;
                }
                if (note.text.rfind("spawn: " + persona.name + " on slots ", 0u) == 0u) {
                    persona_spawned = true;
                }
            }
        }
        if (preset_confirmed && persona_installed && !spawn_sent) {
            if (!lifecycle.send(0u, "spawn " + persona.name)) return 1;
            spawn_sent = true;
            std::printf("live client: imported %s and sent its spawn line\n",
                        persona.name.c_str());
        }
        unsigned persona_agent = ~0u;
        if (persona_spawned && replica != nullptr) {
            for (const auto &agent : replica->agents()) {
                if (agent.id != 0u) persona_agent = agent.id;
            }
        }
        if (persona_agent != ~0u && !persona_said) {
            const std::string line = "task " + std::to_string(persona_agent) +
                                     " Reply with exactly five short words.";
            if (!lifecycle.send(0u, line)) return 1;
            persona_said = true;
            std::printf("live client: spawned persona agent %u and sent its reply request\n",
                        persona_agent);
        }
        if (persona_said && replica != nullptr && persona_agent != ~0u) {
            for (const auto &agent : replica->agents()) {
                if (agent.id != persona_agent) continue;
                for (const auto &event : agent.transcript) {
                    if (event.kind != "reply") continue;
                    persona_replied = true;
                    unsigned joined = 0u;
                    for (const auto &token : replica->tokens()) {
                        if (token.agent == persona_agent && token.turn == event.turn) ++joined;
                    }
                    std::string text;
                    for (const std::string &piece : event.token_text) text += piece;
                    if (!tokens_joined && joined == event.token_text.size() && joined != 0u &&
                        text == event.text) {
                        tokens_joined = true;
                        std::printf("live client: tokens.jsonl joined %u tokens to persona reply: %s\n",
                                    joined, event.text.c_str());
                    }
                }
            }
        }
        if (tokens_joined && !said) {
            if (!lifecycle.send(0u, "say Write the integers from one through five hundred.")) {
                std::fprintf(stderr, "live client: %s\n", lifecycle.refusal().c_str());
                return 1;
            }
            said = true;
            std::puts("live client: sent the long reply request");
        }
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
        if (advanced && reply_stopped && tokens_joined && !stopping) {
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
                 "live client: timeout running=%d preset=%d imported=%d spawned=%d "
                 "persona_reply=%d joined=%d said=%d stop=%d done=%d advanced=%d stopping=%d\n",
                 running ? 1 : 0, preset_confirmed ? 1 : 0, persona_installed ? 1 : 0,
                 persona_spawned ? 1 : 0, persona_replied ? 1 : 0, tokens_joined ? 1 : 0,
                 said ? 1 : 0, stop_sent ? 1 : 0,
                 reply_stopped ? 1 : 0, advanced ? 1 : 0, stopping ? 1 : 0);
    return 1;
}
