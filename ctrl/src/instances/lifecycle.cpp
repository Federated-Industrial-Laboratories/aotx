// Purpose: Create settings and manage headless boot child processes.
// Owns: Child identities, socket clients, and instance result lines.
// Launch shape: One interface thread polls each child without blocking.
// Lifetime: Owned children receive SIGTERM and are reaped at shutdown.
#include "instances/lifecycle.hpp"

#include "client/client.hpp"
#include "process/child.hpp"
#include "replica/replica.hpp"
#include "replica/schema.hpp"

#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <array>
#include <chrono>
#include <fstream>
#include <utility>

namespace aotx::ctrl::instances {
namespace {

struct Held {
    LiveInstance view;
    std::unique_ptr<replica::State> replica;
    std::unique_ptr<client::Client> client;
    bool stop_requested = false;
    bool doomed = false;
    bool caught_up = false;
    std::chrono::steady_clock::time_point stop_deadline{};
};

std::string phase_at(const std::filesystem::path &journal)
{
    std::ifstream file(journal / "phase");
    std::array<char, 8193> line{};
    std::string word;
    if (file.getline(line.data(), static_cast<std::streamsize>(line.size())) &&
        replica::schema::phase(line.data(), word)) return word;
    return "unknown";
}

/* Readers resolve a relative path against the settings file, so every written path is
 * absolute. */
std::filesystem::path settled(const std::filesystem::path &given)
{
    std::error_code error;
    const std::filesystem::path whole = std::filesystem::absolute(given, error);
    return (error ? given : whole).lexically_normal();
}

bool write_settings(const Definition &definition)
{
    std::ofstream file(definition.settings, std::ios::trunc);
    if (!file) return false;
    file << "journal.dir = " << settled(definition.journal).string() << '\n'
         << "models.dir = " << settled(definition.models).string() << '\n'
         << "models.roles = " << definition.roles << '\n'         << "derive.list = console,note,bus,bulk,sequence,requests,transcript\n";
    if (!definition.tools.empty()) {
        file << "tools.root = " << settled(definition.tools).string() << '\n';
    }
    file
         << "window.on = 0\n"
         << "tui.on = 0\n";
    file.flush();
    return file.good();
}

} // namespace

struct Lifecycle::Impl {
    std::vector<Held> held;
    std::vector<std::string> results;
    std::string refusal;
    std::size_t selected = 0u;

    ~Impl()
    {
        for (Held &item : held) {
            if (!item.view.owned || item.view.process < 0) continue;
            const process::End ended = process::end_child(item.view.process);
            item.view.result = ended == process::End::kill
                ? item.view.definition.name + " received SIGKILL after the bounded wait."
                : item.view.definition.name + " ended after SIGTERM.";
            item.view.process = -1;
        }
    }

    void result(Held &item, std::string text)
    {
        item.view.result = std::move(text);
        results.push_back(item.view.result);
    }
};

Lifecycle::Lifecycle() : impl_(std::make_unique<Impl>()) {}
Lifecycle::~Lifecycle() = default;

bool Lifecycle::seed(Definition definition)
{
    Held made;
    made.view.definition = std::move(definition);
    made.view.phase = phase_at(made.view.definition.journal);
    made.replica = std::make_unique<replica::State>(made.view.definition.journal,
                                                    made.view.definition.settings);
    if (!made.replica->open()) {
        impl_->refusal = "The instance binding was refused because the replica does not open.";
        return false;
    }
    made.client = std::make_unique<client::Client>(made.view.definition.journal);
    made.view.connection = made.client->connection();
    made.view.state = made.view.phase == "closed" ? LiveState::stopped : LiveState::attaching;
    made.view.result = "The local instance is bound to the selected journal.";
    impl_->held.push_back(std::move(made));
    return true;
}

bool Lifecycle::create(Definition definition)
{
    impl_->refusal.clear();
    if (definition.name.empty() || definition.journal.empty() || definition.settings.empty() ||
        definition.build.empty() || definition.models.empty()) {
        impl_->refusal = "The instance creation was refused because a required value is empty.";
        return false;
    }
    for (const Held &item : impl_->held) {
        if (item.view.definition.name == definition.name) {
            impl_->refusal = "The instance creation was refused because the name already exists.";
            return false;
        }
    }
    std::error_code error;
    std::filesystem::create_directories(definition.journal, error);
    if (error) {
        impl_->refusal = "The instance creation was refused because the journal does not create.";
        return false;
    }
    if (!definition.settings.parent_path().empty()) {
        std::filesystem::create_directories(definition.settings.parent_path(), error);
    }
    if (error || !write_settings(definition)) {
        impl_->refusal = "The instance creation was refused because the settings do not write.";
        return false;
    }
    Held made;
    made.view.definition = std::move(definition);
    made.replica = std::make_unique<replica::State>(made.view.definition.journal,
                                                    made.view.definition.settings);
    if (!made.replica->open()) {
        impl_->refusal = "The instance creation was refused because the replica does not open.";
        return false;
    }
    made.client = std::make_unique<client::Client>(made.view.definition.journal);
    made.view.result = made.view.definition.name + " was created.";
    impl_->held.push_back(std::move(made));
    impl_->results.push_back(impl_->held.back().view.result);
    return true;
}

bool Lifecycle::remove(std::size_t index)
{
    impl_->refusal.clear();
    if (index >= impl_->held.size()) {
        impl_->refusal = "The instance removal was refused because the instance does not exist.";
        return false;
    }
    if (impl_->held.size() == 1u) {
        impl_->refusal = "The instance removal was refused because one instance must remain.";
        return false;
    }
    Held &item = impl_->held[index];
    if (item.view.owned && item.view.process >= 0) {
        const process::End ended = process::end_child(item.view.process);
        impl_->results.push_back(ended == process::End::kill
            ? item.view.definition.name + " received SIGKILL after the bounded wait."
            : item.view.definition.name + " ended after SIGTERM.");
    }
    /* The erase waits for the next tick, so the panel references of this frame stay valid. */
    item.doomed = true;
    impl_->results.push_back("The instance was removed.");
    return true;
}

bool Lifecycle::start(std::size_t index)
{
    impl_->refusal.clear();
    if (index >= impl_->held.size()) {
        impl_->refusal = "The instance start was refused because the instance does not exist.";
        return false;
    }
    Held &item = impl_->held[index];
    item.view.phase = phase_at(item.view.definition.journal);
    if (item.client) item.client->tick(0.0);
    item.view.connection = item.client ? item.client->connection() : "not connected";
    if (item.view.phase == "running") {
        impl_->refusal = "The instance start was refused because its phase is running.";
        return false;
    }
    if (item.view.connection == "connected") {
        impl_->refusal = "The instance start was refused because its socket answers.";
        return false;
    }
    if (item.view.process >= 0) {
        impl_->refusal = "The instance start was refused because the child is active.";
        return false;
    }
    const std::filesystem::path boot = item.view.definition.build / "aotx_boot";
    if (!std::filesystem::is_regular_file(boot)) {
        impl_->refusal = "The instance start was refused because aotx_boot is not in the build.";
        return false;
    }
    if (!std::filesystem::is_regular_file(item.view.definition.settings)) {
        std::error_code error;
        if (!item.view.definition.settings.parent_path().empty()) {
            std::filesystem::create_directories(item.view.definition.settings.parent_path(),
                                                error);
        }
        if (error || !write_settings(item.view.definition)) {
            impl_->refusal = "The instance start was refused because the settings do not write.";
            return false;
        }
        impl_->result(item, item.view.definition.name + " settings were written.");
    }
    const pid_t child = fork();
    if (child < 0) {
        impl_->refusal = "The instance start was refused because the child does not start.";
        return false;
    }
    if (child == 0) {
        const std::string card = std::to_string(item.view.definition.card);
        setenv("CUDA_VISIBLE_DEVICES", card.c_str(), 1);
        execl(boot.c_str(), boot.c_str(), "--settings", item.view.definition.settings.c_str(),
              static_cast<char *>(nullptr));
        _exit(127);
    }
    item.view.process = child;
    item.view.owned = true;
    item.view.state = LiveState::attaching;
    item.view.phase = "unknown";
    item.view.connection = "not connected";
    item.stop_requested = false;
    if (!item.client) {
        item.client = std::make_unique<client::Client>(item.view.definition.journal);
    }
    impl_->result(item, item.view.definition.name + " child started.");
    return true;
}

bool Lifecycle::stop(std::size_t index)
{
    impl_->refusal.clear();
    if (index >= impl_->held.size() || impl_->held[index].view.process < 0) {
        impl_->refusal = "The instance stop was refused because no owned child is active.";
        return false;
    }
    Held &item = impl_->held[index];
    if (kill(item.view.process, SIGTERM) != 0) {
        impl_->refusal = "The instance stop was refused because SIGTERM was not sent.";
        return false;
    }
    item.stop_requested = true;
    item.stop_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
    impl_->result(item, item.view.definition.name + " received SIGTERM.");
    return true;
}

bool Lifecycle::send(std::size_t index, const std::string &line)
{
    impl_->refusal.clear();
    if (index >= impl_->held.size() || !impl_->held[index].client ||
        !impl_->held[index].client->send_line(line)) {
        impl_->refusal = "The line was refused because the instance connection is not ready.";
        return false;
    }
    impl_->result(impl_->held[index], "The instance accepted " + line + ".");
    return true;
}

int Lifecycle::mirror_descriptor(std::size_t index) const
{
    return index < impl_->held.size() && impl_->held[index].client
        ? impl_->held[index].client->mirror_descriptor() : -1;
}

bool Lifecycle::select(std::size_t index)
{
    if (index >= impl_->held.size()) return false;
    impl_->selected = index;
    return true;
}

std::size_t Lifecycle::selected() const { return impl_->selected; }

replica::State *Lifecycle::replica(std::size_t index)
{
    return index < impl_->held.size() && !impl_->held[index].doomed
        ? impl_->held[index].replica.get() : nullptr;
}

client::Client *Lifecycle::client(std::size_t index)
{
    return index < impl_->held.size() && !impl_->held[index].doomed
        ? impl_->held[index].client.get() : nullptr;
}

void Lifecycle::tick(double now)
{
    for (std::size_t at = impl_->held.size(); at > 0u; --at) {
        if (!impl_->held[at - 1u].doomed) continue;
        impl_->held.erase(impl_->held.begin() + static_cast<std::ptrdiff_t>(at - 1u));
    }
    if (impl_->held.empty()) impl_->selected = 0u;
    else if (impl_->selected >= impl_->held.size()) impl_->selected = impl_->held.size() - 1u;
    for (Held &item : impl_->held) {
        if (item.replica) {
            item.replica->tick(now);
            std::vector<std::string> lines = item.replica->take_results();
            /* The first drain carries the history of the journal; only later lines toast. */
            if (item.caught_up) {
                for (std::string &line : lines) impl_->results.push_back(std::move(line));
            } else {
                item.caught_up = true;
            }
        }
        if (item.client) {
            item.client->tick(now);
            item.view.connection = item.client->connection();
            for (std::string &line : item.client->take_results()) {
                impl_->result(item, std::move(line));
            }
        }
        item.view.phase = phase_at(item.view.definition.journal);
        if (item.view.phase == "running" && item.view.connection == "connected") {
            item.view.state = LiveState::running;
        } else if (item.view.phase == "closed") {
            item.view.state = LiveState::stopped;
        } else if (item.view.process >= 0 || item.view.connection == "attaching") {
            item.view.state = LiveState::attaching;
        }
        if (!item.view.owned || item.view.process < 0) continue;
        int status = 0;
        const pid_t ended = waitpid(item.view.process, &status, WNOHANG);
        if (ended == 0 && item.stop_requested &&
            std::chrono::steady_clock::now() >= item.stop_deadline) {
            if (kill(item.view.process, SIGKILL) == 0 || errno == ESRCH) {
                impl_->result(item, item.view.definition.name +
                                    " received SIGKILL after the bounded wait.");
            }
            continue;
        }
        if (ended <= 0) continue;
        item.view.process = -1;
        item.view.phase = phase_at(item.view.definition.journal);
        item.view.state = LiveState::stopped;
        if (item.stop_requested && item.view.phase == "closed") {
            impl_->result(item, item.view.definition.name + " stopped and wrote the closed phase.");
        } else if (WIFEXITED(status)) {
            impl_->result(item, item.view.definition.name + " child died with status " +
                                      std::to_string(WEXITSTATUS(status)) + ".");
        } else if (WIFSIGNALED(status)) {
            impl_->result(item, item.view.definition.name + " child died from signal " +
                                      std::to_string(WTERMSIG(status)) + ".");
        } else {
            impl_->result(item, item.view.definition.name + " child died.");
        }
    }
}

std::vector<std::string> Lifecycle::take_results()
{
    std::vector<std::string> out;
    out.swap(impl_->results);
    return out;
}

std::vector<LiveInstance> Lifecycle::instances() const
{
    std::vector<LiveInstance> out;
    for (const Held &item : impl_->held) out.push_back(item.view);
    return out;
}

const std::string &Lifecycle::refusal() const { return impl_->refusal; }

const char *state_name(LiveState state)
{
    switch (state) {
    case LiveState::attaching: return "attaching";
    case LiveState::running: return "running";
    case LiveState::stopped: return "stopped";
    }
    return "stopped";
}

} // namespace aotx::ctrl::instances
