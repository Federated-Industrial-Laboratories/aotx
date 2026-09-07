// Purpose: Drive all six first-run pages through one live boot and first reply.
// Owns: Page gate assertions, repeated-action checks, and boot shutdown.
// Launch shape: One host process polls one lifecycle and its replica.
// Lifetime: The lifecycle always ends its owned child before process exit.
#include "instances/lifecycle.hpp"
#include "replica/replica.hpp"
#include "replica/store.hpp"
#include "wizard/gates.hpp"
#include "wizard/models.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

using aotx::ctrl::wizard::ActionLatch;
using aotx::ctrl::wizard::GateFacts;
using aotx::ctrl::wizard::Page;

int fail(const char *reason)
{
    std::fprintf(stderr, "wizard walk: %s\n", reason);
    return 1;
}

bool run_detect(const std::filesystem::path &script)
{
    const pid_t child = fork();
    if (child == 0) {
        execl(script.c_str(), script.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    if (child < 0) return false;
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno != EINTR) return false;
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

std::size_t replies(const aotx::ctrl::replica::State *state, std::string &last)
{
    std::size_t count = 0u;
    if (state == nullptr) return count;
    for (const aotx::ctrl::replica::Agent &agent : state->agents()) {
        for (const aotx::ctrl::replica::TranscriptEvent &event : agent.transcript) {
            if (event.kind != "reply") continue;
            ++count;
            last = event.text;
        }
    }
    return count;
}

bool click_once(ActionLatch &action)
{
    if (!action.begin() || action.begin()) return false;
    return true;
}

struct Options {
    std::string model, role;
    unsigned card = 0;
};

bool options(int argc, char **argv, Options &out)
{
    bool have_model = false, have_role = false, have_card = false;
    for (int index = 4; index < argc; index += 2) {
        if (index + 1 >= argc || argv[index + 1][0] == '\0') return false;
        const std::string name = argv[index], value = argv[index + 1];
        if (name == "--model" && !have_model) { out.model = value; have_model = true; }
        else if (name == "--role" && !have_role && aotx::ctrl::wizard::language_role(value)) {
            out.role = value;
            have_role = true;
        } else if (name == "--card" && !have_card) {
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), out.card);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) return false;
            have_card = true;
        } else return false;
    }
    return true;
}

} // namespace

int main(int argc, char **argv)
{
    Options option;
    if (argc < 4 || !options(argc, argv, option)) {
        std::fputs("usage: aotx_ctrl_wizard_walk <boot> <models> <run-directory> "
                   "[--model <name-or-identity>] [--role <language|language-q4>] [--card <index>]\n", stderr);
        return 2;
    }
    std::error_code error;
    const std::filesystem::path boot = std::filesystem::canonical(argv[1], error);
    if (error) return fail("the boot path does not resolve");
    const std::filesystem::path models = std::filesystem::canonical(argv[2], error);
    if (error) return fail("the model path does not resolve");
    const std::filesystem::path run = std::filesystem::absolute(argv[3], error);
    if (error) return fail("the run path does not resolve");
    GateFacts facts;

    std::printf("wizard walk: page Detect gate %d\n",
                aotx::ctrl::wizard::gate_open(Page::detect, facts) ? 1 : 0);
    ActionLatch detect;
    if (!click_once(detect)) return fail("the Detect action accepted two clicks");
    facts.detected = run_detect(boot.parent_path().parent_path() / "tools/profile-detect.sh");
    detect.complete(facts.detected);
    if (!aotx::ctrl::wizard::gate_open(Page::detect, facts)) {
        return fail("the Detect gate did not open from the card result");
    }
    std::puts("wizard walk: page Detect gate 1; repeated click inert");

    facts.build_ready = std::filesystem::is_regular_file(boot) &&
                        std::filesystem::is_regular_file(boot.parent_path() / "aotx_models");
    if (!aotx::ctrl::wizard::gate_open(Page::build, facts)) {
        return fail("the Build gate did not open from the program files");
    }
    std::puts("wizard walk: page Build gate 1 from program files");

    std::string store_reason;
    std::vector<aotx::ctrl::replica::Model> catalog;
    facts.catalog_read = aotx::ctrl::replica::store::read(
        AOTX_CTRL_MODEL_CATALOG, models, catalog, store_reason);
    aotx::ctrl::wizard::ModelSelection selection;
    selection.refresh(facts.catalog_read, std::move(catalog), store_reason);
    if (!selection.request(option.model, option.role, store_reason)) return fail(store_reason.c_str());
    const aotx::ctrl::replica::Model *selected = selection.selected();
    facts.model_on_disk = selected != nullptr && selected->on_disk;
    if (!aotx::ctrl::wizard::gate_open(Page::model, facts)) {
        return fail("the Model gate did not open from the catalog");
    }
    ActionLatch model_action;
    if (!click_once(model_action)) return fail("the Model action accepted two clicks");
    model_action.complete(true);
    std::puts("wizard walk: page Model gate 1; repeated click inert");

    const aotx::ctrl::wizard::ModelRoles roles = aotx::ctrl::wizard::model_roles(selection);
    if (!roles.ready()) return fail(roles.refusal.c_str());
    facts.model_active = roles.ready();
    if (!aotx::ctrl::wizard::gate_open(Page::activate, facts)) {
        return fail("the Activate gate did not open from the manifest");
    }
    ActionLatch activate;
    if (!click_once(activate)) return fail("the Activate action accepted two clicks");
    activate.complete(true);
    std::puts("wizard walk: page Activate gate 1; repeated click inert");

    aotx::ctrl::instances::Lifecycle lifecycle;
    aotx::ctrl::instances::Definition definition;
    definition.name = "Wizard walk instance";
    definition.journal = run / "journal";
    definition.settings = run / "instance.settings";
    definition.build = boot.parent_path();
    definition.models = models;
    definition.roles = roles.roles;
    definition.card = option.card;
    std::printf("wizard walk: selected %s; role %s; startup roles %s; card %u\n",
                selected->name.c_str(), selected->role.c_str(), definition.roles.c_str(), definition.card);
    if (!roles.notice.empty()) std::printf("wizard walk: %s\n", roles.notice.c_str());
    if (!lifecycle.create(definition)) return fail(lifecycle.refusal().c_str());
    ActionLatch start_action;
    if (!click_once(start_action)) return fail("the Start action accepted two clicks");
    if (!lifecycle.start(0u)) return fail(lifecycle.refusal().c_str());
    if (lifecycle.start(0u) || lifecycle.refusal().empty()) {
        return fail("the lifecycle accepted a second Start action");
    }
    std::puts("wizard walk: page Start gate 0; repeated click inert");

    const auto started = std::chrono::steady_clock::now();
    std::vector<std::string> phases;
    bool running = false;
    while (std::chrono::steady_clock::now() - started < std::chrono::minutes(3)) {
        const double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        lifecycle.tick(now);
        const std::vector<aotx::ctrl::instances::LiveInstance> items = lifecycle.instances();
        if (items.empty()) return fail("the started instance left the lifecycle");
        const auto &instance = items.front();
        if ((instance.phase == "placing" || instance.phase == "replaying" ||
             instance.phase == "running") &&
            (phases.empty() || phases.back() != instance.phase)) {
            phases.push_back(instance.phase);
            std::printf("wizard walk: Start phase %s\n", instance.phase.c_str());
        }
        facts.phase = instance.phase;
        facts.connection = instance.connection;
        const bool gate = aotx::ctrl::wizard::gate_open(Page::start, facts);
        if (gate != aotx::ctrl::wizard::start_gate_open(instance)) {
            return fail("the Start gate differs from the lifecycle state");
        }
        if (gate) {
            running = true;
            start_action.complete(true);
            break;
        }
        if (instance.process < 0) return fail(instance.result.c_str());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!running) return fail("the Start gate did not open");
    if (phases.size() < 2u || phases.front() != "placing" || phases.back() != "running") {
        return fail("the Start phase transitions were incomplete");
    }
    std::puts("wizard walk: page Start gate 1 from running phase and socket");

    std::string reply;
    const std::size_t before = replies(lifecycle.replica(0u), reply);
    ActionLatch first_say;
    if (!click_once(first_say)) return fail("the First say action accepted two clicks");
    if (!lifecycle.send(0u, "say Hello.")) return fail(lifecycle.refusal().c_str());
    std::puts("wizard walk: page First say gate 0; repeated click inert");
    bool answered = false;
    while (std::chrono::steady_clock::now() - started < std::chrono::minutes(3)) {
        const double now = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        lifecycle.tick(now);
        facts.reply_received = replies(lifecycle.replica(0u), reply) > before;
        if (aotx::ctrl::wizard::gate_open(Page::first_say, facts)) {
            answered = true;
            first_say.complete(true);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (!answered) return fail("the First say gate did not receive a reply");
    if (!facts.reply_received || reply.empty()) {
        return fail("the First say gate opened without a completed reply");
    }
    std::printf("wizard walk: page First say gate 1; reply %s\n", reply.c_str());

    if (!lifecycle.stop(0u)) return fail(lifecycle.refusal().c_str());
    const auto stopping = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - stopping < std::chrono::seconds(10)) {
        lifecycle.tick(180.0);
        const auto items = lifecycle.instances();
        if (!items.empty() && items[0].phase == "closed" && items[0].process < 0) {
            std::puts("wizard walk: boot ended in the closed phase");
            return 0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return fail("the boot did not end in the closed phase");
}
