// Purpose: Drive live panel lines and confirm their typed replica results.
// Owns: One replica reader and one attach socket client.
// Launch shape: One process polls one live system and three panel actions.
// Lifetime: The process ends after all results arrive or a timeout occurs.
#include "client/client.hpp"
#include "replica/replica.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::fputs("usage: aotx_ctrl_live_client <journal> <settings> <skill-directory>\n",
                   stderr);
        return 2;
    }
    const std::filesystem::path skill = argv[3];
    const std::string skill_name = skill.filename().string();
    aotx::ctrl::replica::State replica(argv[1], argv[2]);
    aotx::ctrl::client::Client client(argv[1]);
    if (!replica.open()) {
        std::fputs("live client: the replica does not open\n", stderr);
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    bool sent = false;
    bool setting_seen = false;
    bool import_seen = false;
    bool module_seen = false;
    bool models_seen = false;
    bool catalog_seen = !replica.models().empty();
    while (std::chrono::steady_clock::now() - start < std::chrono::minutes(2)) {
        const double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        replica.tick(now);
        client.tick(now);
        for (const std::string &result : replica.take_results()) {
            std::printf("live client: replica result: %s\n", result.c_str());
            if (result == "setting decode.reply_limit 37") setting_seen = true;
            if (result.rfind("module " + skill_name + " skill import ", 0u) == 0u) {
                import_seen = true;
            }
        }
        for (const std::string &result : client.take_results()) {
            std::printf("live client: socket result: %s\n", result.c_str());
        }
        if (!sent && std::string(client.connection()) == "connected") {
            std::printf("live client: connected, mirror descriptor %d\n",
                        client.mirror_descriptor());
            if (!client.send_line("set decode.reply_limit 37") ||
                !client.send_line("import " + skill.string()) ||
                !client.send_line("models")) return 2;
            sent = true;
            std::puts("live client: sent set decode.reply_limit 37");
            std::printf("live client: sent import %s\n", skill.c_str());
            std::puts("live client: sent models");
        }
        for (const aotx::ctrl::replica::Module &module : replica.modules()) {
            if (module.name == skill_name && module.kind == "skill") module_seen = true;
        }
        for (const std::string &line : replica.console()) {
            if (line == "models: role file sha256 tick") models_seen = true;
        }
        catalog_seen = catalog_seen || !replica.models().empty();
        if (sent && setting_seen && import_seen && module_seen && models_seen && catalog_seen) {
            std::puts("live client: the setting acknowledgment reached the parser");
            std::puts("live client: the imported skill reached modules.jsonl and the panel model");
            std::printf("live client: the model catalog contains %zu entries\n",
                        replica.models().size());
            std::puts("live client: the models command reached the console replica");
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::fprintf(stderr,
                 "live client: timeout sent=%d setting=%d import=%d module=%d models=%d catalog=%d\n",
                 sent ? 1 : 0, setting_seen ? 1 : 0, import_seen ? 1 : 0,
                 module_seen ? 1 : 0, models_seen ? 1 : 0, catalog_seen ? 1 : 0);
    return 1;
}
