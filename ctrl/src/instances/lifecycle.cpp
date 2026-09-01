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

#include <fcntl.h>

#include <cerrno>
#include <array>
#include <chrono>
#include <iterator>
#include <fstream>
#include <utility>

namespace aotx::ctrl::instances {
namespace {

struct Held {
    LiveInstance view;
    std::unique_ptr<replica::State> replica;
    std::unique_ptr<client::Client> client;
    bool stop_requested = false;
    bool kill_sent = false;
    bool doomed = false;
    bool registered = false;
    bool caught_up = false;
    int child_out = -1;
    std::string child_partial;
    std::string child_last;
    std::string reported_phase;
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
         << "models.roles = " << definition.roles << '\n'
         << "derive.list = console,note,bus,bulk,sequence,requests,transcript,tokens,pages\n";
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
    std::filesystem::path registry;

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

    /* One quoted JSON string; a path with a quote or a backslash stays whole. */
    static std::string field(const std::string &text)
    {
        std::string out = "\"";
        for (char byte : text) {
            if (byte == '"' || byte == '\\') out += '\\';
            out += byte;
        }
        out += '"';
        return out;
    }

    /* The registry keeps every created instance, so a restart lists them again. */
    void save_registry()
    {
        if (registry.empty()) return;
        std::error_code error;
        if (!registry.parent_path().empty()) {
            std::filesystem::create_directories(registry.parent_path(), error);
        }
        std::ofstream file(registry, std::ios::trunc);
        for (const Held &item : held) {
            if (!item.registered || item.doomed) continue;
            const Definition &definition = item.view.definition;
            file << "{\"name\":" << field(definition.name)
                 << ",\"journal\":" << field(settled(definition.journal).string())
                 << ",\"settings\":" << field(settled(definition.settings).string())
                 << ",\"build\":" << field(settled(definition.build).string())
                 << ",\"models\":" << field(settled(definition.models).string())
                 << ",\"tools\":" << field(definition.tools.empty()
                                                 ? std::string()
                                                 : settled(definition.tools).string())
                 << ",\"card\":" << definition.card;
            for (const auto &name : definition.conversation_names) {
                file << ",\"conversation_" << name.first << "\":" << field(name.second);
            }
            file << "}\n";
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

void Lifecycle::set_registry(std::filesystem::path file)
{
    impl_->registry = std::move(file);
}

bool Lifecycle::seed(Definition definition, bool registered)
{
    Held made;
    made.view.definition = std::move(definition);
    made.view.phase = phase_at(made.view.definition.journal);
    made.reported_phase = made.view.phase;
    made.replica = std::make_unique<replica::State>(made.view.definition.journal,
                                                    made.view.definition.settings);
    if (!made.replica->open()) {
        impl_->refusal = "The instance binding was refused because the replica does not open.";
        return false;
    }
    made.client = std::make_unique<client::Client>(made.view.definition.journal);
    made.view.connection = made.client->connection();
    made.view.state = made.view.phase == "closed" ? LiveState::stopped : LiveState::attaching;
    made.view.result = made.view.definition.name + " is bound to its journal.";
    made.registered = registered;
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
    made.registered = true;
    impl_->held.push_back(std::move(made));
    impl_->results.push_back(impl_->held.back().view.result);
    impl_->save_registry();
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
    if (item.view.phase == "placing" || item.view.phase == "replaying") {
        std::error_code fresh_error;
        const auto stamp = std::filesystem::last_write_time(
            item.view.definition.journal / "phase", fresh_error);
        if (!fresh_error && std::filesystem::file_time_type::clock::now() - stamp <
                                std::chrono::seconds(30)) {
            impl_->refusal = "The instance start was refused because a boot is under way.";
            return false;
        }
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
    } else if (!item.view.definition.tools.empty()) {
        std::ifstream reading(item.view.definition.settings);
        const std::string content((std::istreambuf_iterator<char>(reading)),
                                  std::istreambuf_iterator<char>());
        if (content.find("tools.root") == std::string::npos) {
            std::ofstream appending(item.view.definition.settings, std::ios::app);
            appending << "tools.root = " << settled(item.view.definition.tools).string()
                      << '\n';
            impl_->result(item, item.view.definition.name + " settings gained the tool root.");
        }
    }
    int report[2] = {-1, -1};
    if (pipe(report) != 0) {
        impl_->refusal = "The instance start was refused because the pipe does not open.";
        return false;
    }
    const pid_t child = fork();
    if (child < 0) {
        close(report[0]);
        close(report[1]);
        impl_->refusal = "The instance start was refused because the child does not start.";
        return false;
    }
    if (child == 0) {
        dup2(report[1], 1);
        dup2(report[1], 2);
        close(report[0]);
        close(report[1]);
        const std::string card = std::to_string(item.view.definition.card);
        setenv("CUDA_VISIBLE_DEVICES", card.c_str(), 1);
        execl(boot.c_str(), boot.c_str(), "--settings", item.view.definition.settings.c_str(),
              static_cast<char *>(nullptr));
        _exit(127);
    }
    close(report[1]);
    if (item.child_out >= 0) close(item.child_out);
    item.child_out = report[0];
    fcntl(item.child_out, F_SETFL, O_NONBLOCK);
    item.child_partial.clear();
    item.child_last.clear();
    item.view.process = child;
    item.view.owned = true;
    item.view.state = LiveState::attaching;
    item.view.phase = "unknown";
    item.reported_phase.clear();
    item.view.connection = "not connected";
    item.stop_requested = false;
    item.kill_sent = false;
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
    item.kill_sent = false;
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

bool Lifecycle::name_conversation(std::size_t index, unsigned agent,
                                  const std::string &name, bool persist)
{
    impl_->refusal.clear();
    if (index >= impl_->held.size() || name.empty() || name.size() > 80u) {
        impl_->refusal = "The conversation name was refused because it is not valid.";
        return false;
    }
    impl_->held[index].view.definition.conversation_names[agent] = name;
    if (replica::State *state = impl_->held[index].replica.get()) {
        for (replica::Agent &row : state->agents()) if (row.id == agent) row.conversation = name;
    }
    if (persist) {
        impl_->save_registry();
        impl_->results.push_back("The conversation name was saved.");
    }
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
    bool erased = false;
    for (std::size_t at = impl_->held.size(); at > 0u; --at) {
        if (!impl_->held[at - 1u].doomed) continue;
        impl_->held.erase(impl_->held.begin() + static_cast<std::ptrdiff_t>(at - 1u));
        erased = true;
    }
    if (erased) impl_->save_registry();
    if (impl_->held.empty()) impl_->selected = 0u;
    else if (impl_->selected >= impl_->held.size()) impl_->selected = impl_->held.size() - 1u;
    for (Held &item : impl_->held) {
        if (item.replica) {
            item.replica->tick(now);
            for (replica::Agent &agent : item.replica->agents()) {
                const auto name = item.view.definition.conversation_names.find(agent.id);
                if (name != item.view.definition.conversation_names.end()) {
                    agent.conversation = name->second;
                }
            }
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
        if (item.view.phase != item.reported_phase &&
            (item.view.phase == "placing" || item.view.phase == "replaying" ||
             item.view.phase == "running")) {
            item.reported_phase = item.view.phase;
            impl_->result(item, item.view.definition.name + " phase is " + item.view.phase + ".");
        }
        if (item.view.phase == "running" && item.view.connection == "connected") {
            item.view.state = LiveState::running;
        } else if (item.view.phase == "closed") {
            item.view.state = LiveState::stopped;
        } else if (item.view.process >= 0 || item.view.connection == "attaching") {
            item.view.state = LiveState::attaching;
        }
        if (item.child_out >= 0) {
            std::array<char, 512> bytes{};
            ssize_t got = 0;
            while ((got = read(item.child_out, bytes.data(), bytes.size())) > 0) {
                item.child_partial.append(bytes.data(), static_cast<std::size_t>(got));
            }
            std::size_t mark = 0u;
            while ((mark = item.child_partial.find('\n')) != std::string::npos) {
                if (mark > 0u) item.child_last = item.child_partial.substr(0u, mark);
                item.child_partial.erase(0u, mark + 1u);
            }
        }
        if (!item.view.owned || item.view.process < 0) continue;
        int status = 0;
        const pid_t ended = waitpid(item.view.process, &status, WNOHANG);
        if (ended == 0 && item.stop_requested && !item.kill_sent &&
            std::chrono::steady_clock::now() >= item.stop_deadline) {
            if (kill(item.view.process, SIGKILL) == 0 || errno == ESRCH) {
                item.kill_sent = true;
                impl_->result(item, item.view.definition.name +
                                    " received SIGKILL after the bounded wait.");
            }
            continue;
        }
        if (ended <= 0) continue;
        item.view.process = -1;
        if (item.child_out >= 0) {
            if (!item.child_partial.empty()) {
                item.child_last = item.child_partial;
                item.child_partial.clear();
            }
            close(item.child_out);
            item.child_out = -1;
        }
        item.view.phase = phase_at(item.view.definition.journal);
        item.view.state = LiveState::stopped;
        if (item.stop_requested && item.view.phase == "closed") {
            impl_->result(item, item.view.definition.name + " stopped and wrote the closed phase.");
        } else if (WIFEXITED(status)) {
            const std::string reason =
                item.child_last.empty() ? "" : ": " + item.child_last;
            impl_->result(item, item.view.definition.name + " child died with status " +
                                      std::to_string(WEXITSTATUS(status)) + reason + ".");
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
