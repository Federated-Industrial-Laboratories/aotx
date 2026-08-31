// Purpose: Send one line and confirm its typed transcript records.
// Owns: One replica reader and one attach socket client.
// Launch shape: One process polls one live system.
// Lifetime: The process ends after one folded reply or a timeout.
#include "client/client.hpp"
#include "replica/replica.hpp"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 3) {
        std::fputs("usage: aotx_ctrl_live_client <journal> <text>\n", stderr);
        return 2;
    }
    const std::string text = argv[2];
    aotx::ctrl::replica::State replica(argv[1], {});
    aotx::ctrl::client::Client client(argv[1]);
    if (!replica.open()) {
        std::fputs("live client: the replica does not open\n", stderr);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    bool sent = false;
    bool line_seen = false;
    bool reply_seen = false;
    bool stream_seen = false;
    std::string reply_text;
    std::vector<std::string> snapshots;
    std::uint64_t part_lines = 0u;
    while (std::chrono::steady_clock::now() - start < std::chrono::minutes(5)) {
        const double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        replica.tick(now);
        client.tick(now);
        for (const std::string &result : replica.take_results()) {
            std::printf("live client: replica result: %s\n", result.c_str());
        }
        for (const std::string &result : client.take_results()) {
            std::printf("live client: socket result: %s\n", result.c_str());
        }
        if (!sent && std::string(client.connection()) == "connected") {
            std::printf("live client: connected, mirror descriptor %d\n",
                        client.mirror_descriptor());
            if (!client.send_line("say " + text)) return 2;
            sent = true;
            std::printf("live client: sent say %s\n", text.c_str());
        }
        const auto &agents = replica.agents();
        if (!agents.empty()) {
            const auto &agent = agents.front();
            const auto &events = agent.transcript;
            snapshots.resize(events.size());
            for (std::size_t index = 0u; index < events.size(); ++index) {
                const auto &event = events[index];
                const std::string snapshot = event.kind + "\n" + event.text + "\n" +
                                             event.status;
                if (snapshots[index] != snapshot) {
                    std::printf("live client: event kind=%s turn=%llu text=%s\n",
                                event.kind.c_str(),
                                static_cast<unsigned long long>(event.turn),
                                event.text.c_str());
                    snapshots[index] = snapshot;
                }
                if (event.kind == "line" && event.text == text) line_seen = true;
                if (event.kind == "part" && !event.text.empty()) stream_seen = true;
                if (event.kind == "reply" && !event.text.empty()) {
                    reply_seen = true;
                    reply_text = event.text;
                }
            }
            if (agent.part_lines != part_lines) {
                part_lines = agent.part_lines;
                std::printf("live client: parsed part lines=%llu folded text=%s\n",
                            static_cast<unsigned long long>(part_lines),
                            agent.folded_reply.c_str());
            }
            if (sent && line_seen && reply_seen) {
                if (!stream_seen || part_lines < 2u) {
                    std::fputs("live client: the part lines did not stream\n", stderr);
                    return 1;
                }
                if (!agent.fold_replaced || reply_text != agent.folded_reply) {
                    std::fputs("live client: the final reply did not replace its parts\n", stderr);
                    return 1;
                }
                std::puts("live client: the sent line matches the replica");
                std::puts("live client: the final reply matches and replaces its streamed parts");
                return 0;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::fprintf(stderr, "live client: timeout sent=%d line=%d part=%d reply=%d\n",
                 sent ? 1 : 0, line_seen ? 1 : 0, stream_seen ? 1 : 0,
                 reply_seen ? 1 : 0);
    return 1;
}
