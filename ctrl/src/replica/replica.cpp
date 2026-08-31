// Purpose: Discover boots and tail the typed disk replica files.
// Owns: File cursors, inotify watches, and parsed live records.
// Launch shape: One interface thread refreshes changed files.
// Lifetime: State exists from program start until program exit.
#include "replica/replica.hpp"

#include "replica/json.hpp"
#include "replica/schema.hpp"
#include "replica/store.hpp"

#include <sys/inotify.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <system_error>

#ifndef AOTX_CTRL_LANGUAGE_ROLE
#define AOTX_CTRL_LANGUAGE_ROLE "language"
#endif
#ifndef AOTX_CTRL_MODEL_CATALOG
#define AOTX_CTRL_MODEL_CATALOG "share/models/catalog.jsonl"
#endif

namespace aotx::ctrl::replica {
namespace {

struct Cursor {
    std::uintmax_t offset = 0u;
    std::uint64_t line = 0u;
    std::string partial;
};

bool boot_name(const std::string &name)
{
    if (name.size() != 16u) return false;
    for (const unsigned char byte : name) if (!std::isxdigit(byte)) return false;
    return true;
}

bool agent_name(const std::filesystem::path &path, unsigned &id)
{
    if (path.extension() != ".jsonl") return false;
    const std::string stem = path.stem().string();
    unsigned parsed = 0u;
    const auto result = std::from_chars(stem.data(), stem.data() + stem.size(), parsed);
    if (stem.empty() || result.ec != std::errc() || result.ptr != stem.data() + stem.size()) {
        return false;
    }
    id = parsed;
    return true;
}

std::string trim(std::string text)
{
    const auto space = [](unsigned char byte) { return byte == ' ' || byte == '\t' || byte == '\r'; };
    while (!text.empty() && space(static_cast<unsigned char>(text.front()))) text.erase(text.begin());
    while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
    return text;
}

template <class Take>
void read_lines(const std::filesystem::path &path, Cursor &cursor, Take take)
{
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(path, error);
    if (error) return;
    if (size < cursor.offset) cursor = Cursor{};
    if (size == cursor.offset) return;
    std::ifstream file(path, std::ios::binary);
    if (!file) return;
    file.seekg(static_cast<std::streamoff>(cursor.offset));
    std::string bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    cursor.offset += bytes.size();
    cursor.partial += bytes;
    std::size_t at = 0u;
    while (true) {
        const std::size_t end = cursor.partial.find('\n', at);
        if (end == std::string::npos) break;
        std::string line = cursor.partial.substr(at, end - at);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        ++cursor.line;
        take(line, cursor.line);
        at = end + 1u;
    }
    cursor.partial.erase(0u, at);
}

void fold_event(Agent &agent, TranscriptEvent event)
{
    if (event.kind == "part") {
        agent.reply_bound = false;
        agent.fold_replaced = false;
        ++agent.part_lines;
        if (!agent.transcript.empty() && agent.transcript.back().kind == "part" &&
            agent.transcript.back().turn == event.turn) {
            agent.transcript.back().text += event.text;
            agent.transcript.back().tick = event.tick;
        } else {
            agent.transcript.push_back(std::move(event));
        }
        agent.folded_reply = agent.transcript.back().text;
        return;
    }
    if (event.kind == "reply") {
        agent.reply_bound = false;
        const bool folded = !agent.transcript.empty() &&
                            agent.transcript.back().kind == "part" &&
                            agent.transcript.back().turn == event.turn;
        agent.fold_replaced = folded && agent.transcript.back().text == event.text;
        if (folded) agent.transcript.back() = std::move(event);
        else agent.transcript.push_back(std::move(event));
        return;
    }
    if (event.kind == "bound") agent.reply_bound = true;
    agent.transcript.push_back(std::move(event));
}

} // namespace

struct State::Impl {
    explicit Impl(std::filesystem::path journal_path, std::filesystem::path settings_path)
        : journal(std::move(journal_path)), settings(std::move(settings_path))
    {
        if (settings.empty()) settings = journal.parent_path() / "aotx.settings";
    }

    std::filesystem::path journal;
    std::filesystem::path settings;
    std::filesystem::path active_boot;
    std::string phase = "unknown";
    std::string language = "No language model";
    std::vector<Boot> boots;
    std::vector<Agent> agents;
    std::vector<Note> notes;
    std::vector<Request> requests;
    std::vector<PendingRequest> pending;
    std::vector<AgentState> agent_states;
    std::vector<Module> modules;
    std::vector<Model> models;
    std::filesystem::path model_directory;
    std::vector<std::string> console;
    std::vector<std::string> results;
    std::map<std::string, Cursor> cursors;
    std::map<std::string, int> watches;
    int notify_fd = -1;
    double next_scan = 0.0;
    std::string last_phase_line;
    std::filesystem::file_time_type model_time{};
    bool model_seen = false;
    bool model_result_seen = false;
    bool initial_read = true;
    std::string model_error;

    ~Impl()
    {
        if (notify_fd >= 0) ::close(notify_fd);
    }

    void refusal(const char *name, std::uint64_t line)
    {
        results.push_back("The " + std::string(name) + " line " + std::to_string(line) +
                          " was refused.");
    }

    void watch(const std::filesystem::path &path)
    {
        if (notify_fd < 0) return;
        const std::string name = path.string();
        if (watches.find(name) != watches.end()) return;
        const int descriptor = inotify_add_watch(
            notify_fd, name.c_str(), IN_CREATE | IN_CLOSE_WRITE | IN_MODIFY | IN_MOVED_FROM |
                                         IN_MOVED_TO | IN_DELETE | IN_DELETE_SELF | IN_MOVE_SELF);
        if (descriptor >= 0) watches.emplace(name, descriptor);
    }

    void drain_notifications()
    {
        if (notify_fd < 0) return;
        std::array<char, 4096> buffer{};
        while (true) {
            const ssize_t count = ::read(notify_fd, buffer.data(), buffer.size());
            if (count > 0) continue;
            if (count < 0 && errno == EINTR) continue;
            break;
        }
    }

    void discover_boots()
    {
        std::error_code error;
        std::vector<std::pair<std::filesystem::file_time_type, Boot>> found;
        for (const auto &entry : std::filesystem::directory_iterator(journal, error)) {
            if (error) break;
            const std::string name = entry.path().filename().string();
            if (!entry.is_directory(error) || error || !boot_name(name)) {
                error.clear();
                continue;
            }
            const auto time = entry.last_write_time(error);
            if (error) {
                error.clear();
                continue;
            }
            found.push_back({time, {name, entry.path()}});
        }
        std::sort(found.begin(), found.end(), [](const auto &left, const auto &right) {
            if (left.first != right.first) return left.first > right.first;
            return left.second.name > right.second.name;
        });
        boots.clear();
        for (const auto &item : found) boots.push_back(item.second);
        const std::filesystem::path newest = boots.empty() ? std::filesystem::path{}
                                                           : boots.front().directory;
        if (newest == active_boot) return;
        active_boot = newest;
        agents.clear();
        console.clear();
        notes.clear();
        requests.clear();
        pending.clear();
        agent_states.clear();
        modules.clear();
        cursors.clear();
        language = "No language model";
        model_seen = false;
        model_result_seen = false;
        if (!active_boot.empty()) {
            results.push_back("The newest boot is " + active_boot.filename().string() + ".");
            watch(active_boot);
            watch(active_boot / "transcript");
        }
    }

    std::filesystem::path setting_path() const
    {
        return settings.empty() ? journal.parent_path() / "aotx.settings" : settings;
    }

    std::filesystem::path models_path() const
    {
        std::string directory;
        if (!setting_value(setting_path(), "models.dir", directory) || directory.empty()) {
            directory = "models";
        }
        std::filesystem::path path(directory);
        if (path.is_relative()) path = setting_path().parent_path() / path;
        return path.lexically_normal();
    }

    void read_phase()
    {
        watch(journal / "phase");
        std::ifstream file(journal / "phase");
        std::string line;
        if (!file || !std::getline(file, line) || line == last_phase_line) return;
        last_phase_line = line;
        std::string word;
        if (!schema::phase(line, word)) {
            refusal("phase", 1u);
            return;
        }
        phase = word;
    }

    Agent *find_agent(unsigned id)
    {
        for (Agent &agent : agents) if (agent.id == id) return &agent;
        return nullptr;
    }

    std::string language_role() const
    {
        std::string roles;
        std::string role = AOTX_CTRL_LANGUAGE_ROLE;
        (void)setting_value(setting_path(), "models.roles", roles);
        std::istringstream role_list(roles);
        std::string item;
        while (std::getline(role_list, item, ',')) {
            item = trim(item);
            if (item == "language" || item == "language-q4") role = item;
        }
        return role;
    }

    void read_transcripts()
    {
        if (active_boot.empty()) return;
        const std::filesystem::path directory = active_boot / "transcript";
        std::error_code error;
        watch(directory);
        for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
            if (error) break;
            unsigned id;
            if (!entry.is_regular_file(error) || error || !agent_name(entry.path(), id)) {
                error.clear();
                continue;
            }
            Agent *agent = find_agent(id);
            if (agent == nullptr) {
                agents.push_back({id, "Agent " + std::to_string(id), {}});
                std::sort(agents.begin(), agents.end(),
                          [](const Agent &left, const Agent &right) { return left.id < right.id; });
                agent = find_agent(id);
            }
            watch(entry.path());
            Cursor &cursor = cursors[entry.path().string()];
            read_lines(entry.path(), cursor, [this, agent](const std::string &line, std::uint64_t at) {
                TranscriptEvent event;
                if (!schema::transcript(line, event)) refusal("transcript", at);
                else fold_event(*agent, std::move(event));
            });
        }
    }

    void read_console()
    {
        if (active_boot.empty()) return;
        const std::filesystem::path path = active_boot / "console.log";
        watch(path);
        read_lines(path, cursors[path.string()], [this](const std::string &line, std::uint64_t) {
            console.push_back(line);
        });
    }

    void read_notes()
    {
        const std::filesystem::path directory = journal / "bus";
        std::error_code error;
        watch(directory);
        for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
            if (error) break;
            if (!entry.is_regular_file(error) || error || entry.path().extension() != ".jsonl") {
                error.clear();
                continue;
            }
            watch(entry.path());
            Cursor &cursor = cursors[entry.path().string()];
            read_lines(entry.path(), cursor, [this](const std::string &line, std::uint64_t at) {
                json::Value value;
                std::string type;
                if (!json::parse(line, value) || !json::text(value, "type", type)) {
                    refusal("note", at);
                    return;
                }
                if (type != "note") return;
                Note note;
                if (!schema::note(line, note)) refusal("note", at);
                else {
                    if (!active_boot.empty() && note.boot != active_boot.filename().string()) {
                        return;
                    }
                    std::string loaded;
                    if (schema::model_load(note.text, language_role(), loaded) &&
                        !loaded.empty()) {
                        language = loaded;
                        model_result_seen = true;
                    }
                    std::string fetch_name;
                    std::string fetch_state;
                    std::uint64_t fetched = 0u;
                    std::uint64_t total = 0u;
                    if (schema::fetch_result(note.text, fetch_name, fetched, total,
                                             fetch_state)) {
                        for (Model &model : models) {
                            if (model.name != fetch_name) continue;
                            model.fetch_result = fetch_state;
                            if (fetch_state == "progress") {
                                model.fetching = true;
                                model.fetched = fetched;
                                model.fetch_total = total;
                            } else if (fetch_state == "started" ||
                                       fetch_state.rfind("host ", 0u) == 0u ||
                                       fetch_state.rfind("restart ", 0u) == 0u) {
                                model.fetching = true;
                            } else {
                                model.fetching = false;
                                if (fetch_state == "on disk") model.on_disk = true;
                            }
                        }
                    }
                    PendingRequest pending_request;
                    if (schema::pending_request(note.text, pending_request)) {
                        const auto same = [&pending_request](const PendingRequest &item) {
                            return item.request == pending_request.request;
                        };
                        if (std::find_if(pending.begin(), pending.end(), same) == pending.end()) {
                            pending.push_back(std::move(pending_request));
                        }
                    }
                    AgentState agent_state;
                    if (schema::agent_state(note.text, agent_state)) {
                        const auto same = [&agent_state](const AgentState &item) {
                            return item.agent == agent_state.agent;
                        };
                        const auto found = std::find_if(agent_states.begin(), agent_states.end(), same);
                        if (agent_state.event == "released") {
                            if (found != agent_states.end()) agent_states.erase(found);
                        } else if (found == agent_states.end()) {
                            agent_states.push_back(std::move(agent_state));
                        } else {
                            *found = std::move(agent_state);
                        }
                    }
                    if (!initial_read && schema::action_result(note.text)) {
                        results.push_back(note.text);
                    }
                    notes.push_back(std::move(note));
                }
            });
        }
    }

    void read_requests()
    {
        const std::filesystem::path path = journal / "requests.jsonl";
        watch(path);
        read_lines(path, cursors[path.string()], [this](const std::string &line, std::uint64_t at) {
            Request request;
            if (!schema::request(line, request)) refusal("request", at);
            else {
                if (request.authorization == "granted") {
                    pending.erase(std::remove_if(pending.begin(), pending.end(),
                        [&request](const PendingRequest &item) {
                            return item.request == request.request;
                        }), pending.end());
                }
                requests.push_back(std::move(request));
            }
        });
    }

    void resolve_pending()
    {
        for (const Agent &agent : agents) {
            for (const TranscriptEvent &event : agent.transcript) {
                if (event.kind != "grant" && event.kind != "refuse") continue;
                pending.erase(std::remove_if(pending.begin(), pending.end(),
                    [&event](const PendingRequest &item) {
                        return item.request == event.request;
                    }), pending.end());
            }
        }
    }

    void read_modules()
    {
        const std::filesystem::path path = journal / "modules.jsonl";
        watch(path);
        read_lines(path, cursors[path.string()], [this](const std::string &line, std::uint64_t at) {
            Module module;
            if (!schema::module(line, module)) refusal("module", at);
            else modules.push_back(std::move(module));
        });
    }

    void read_model()
    {
        const std::string role = language_role();
        const std::filesystem::path path = models_path() / "manifest.jsonl";
        watch(path);
        std::error_code error;
        const auto changed = std::filesystem::last_write_time(path, error);
        if (error || (model_seen && changed == model_time)) return;
        model_time = changed;
        model_seen = true;
        std::ifstream file(path);
        if (!file) return;
        std::string line;
        std::string active;
        std::uint64_t at = 0u;
        while (std::getline(file, line)) {
            ++at;
            if (trim(line).empty()) continue;
            std::string name;
            if (!schema::language_model(line, role, name)) {
                refusal("model manifest", at);
                continue;
            }
            if (!name.empty()) active = name;
        }
        if (!model_result_seen) language = active.empty() ? "No language model" : active;
    }

    void read_model_store()
    {
        model_directory = models_path();
        watch(model_directory);
        watch(model_directory / "store.jsonl");
        watch(model_directory / "manifest.jsonl");
        std::string reason;
        if (!store::read(AOTX_CTRL_MODEL_CATALOG, model_directory, models, reason)) {
            if (reason != model_error) results.push_back(reason);
            model_error = reason;
        } else {
            model_error.clear();
        }
    }

    void refresh()
    {
        discover_boots();
        read_phase();
        read_transcripts();
        read_console();
        read_model_store();
        read_notes();
        read_requests();
        resolve_pending();
        read_modules();
        read_model();
    }
};

State::State(std::filesystem::path journal, std::filesystem::path settings)
    : impl_(std::make_unique<Impl>(std::move(journal), std::move(settings))) {}

State::~State() = default;

bool State::open()
{
    std::error_code error;
    if (impl_->journal.empty() || !std::filesystem::is_directory(impl_->journal, error) || error) {
        impl_->results.push_back("The journal directory does not read.");
        return false;
    }
    impl_->notify_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
    if (impl_->notify_fd < 0) {
        impl_->results.push_back("The directory watch does not start.");
        return false;
    }
    impl_->watch(impl_->journal);
    impl_->watch(impl_->journal / "bus");
    impl_->refresh();
    impl_->initial_read = false;
    return true;
}

void State::tick(double now)
{
    impl_->drain_notifications();
    if (now < impl_->next_scan) return;
    impl_->next_scan = now + 0.05;
    impl_->refresh();
}

std::vector<std::string> State::take_results()
{
    std::vector<std::string> out;
    out.swap(impl_->results);
    return out;
}

std::size_t State::create_conversation()
{
    return impl_->agents.size();
}

const std::filesystem::path &State::journal() const { return impl_->journal; }
const std::filesystem::path &State::settings() const { return impl_->settings; }
const std::string &State::phase() const { return impl_->phase; }
const std::string &State::language_model() const { return impl_->language; }
const std::vector<Boot> &State::boots() const { return impl_->boots; }
std::vector<Agent> &State::agents() { return impl_->agents; }
const std::vector<Agent> &State::agents() const { return impl_->agents; }
const std::vector<Note> &State::notes() const { return impl_->notes; }
const std::vector<Request> &State::requests() const { return impl_->requests; }
const std::vector<PendingRequest> &State::pending_requests() const { return impl_->pending; }
const std::vector<AgentState> &State::agent_states() const { return impl_->agent_states; }
const std::vector<Module> &State::modules() const { return impl_->modules; }
const std::vector<Model> &State::models() const { return impl_->models; }
const std::filesystem::path &State::models_directory() const { return impl_->model_directory; }
const std::vector<std::string> &State::console() const { return impl_->console; }

bool setting_value(const std::filesystem::path &path, const std::string &key,
                   std::string &value)
{
    std::ifstream file(path);
    std::string line;
    bool found = false;
    if (!file) return false;
    while (std::getline(file, line)) {
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const std::size_t equal = line.find('=');
        if (equal == std::string::npos) continue;
        const std::string name = trim(line.substr(0u, equal));
        if (name != key) continue;
        value = trim(line.substr(equal + 1u));
        found = true;
    }
    return found;
}

bool read_boot_transcripts(const std::filesystem::path &boot, std::vector<Agent> &agents,
                           std::string &reason)
{
    agents.clear();
    std::error_code error;
    const std::filesystem::path directory = boot / "transcript";
    for (const auto &entry : std::filesystem::directory_iterator(directory, error)) {
        if (error) break;
        unsigned id = 0u;
        if (!entry.is_regular_file(error) || error || !agent_name(entry.path(), id)) {
            error.clear();
            continue;
        }
        Agent made{id, "Agent " + std::to_string(id), {}};
        std::ifstream file(entry.path());
        std::string line;
        std::uint64_t at = 0u;
        while (std::getline(file, line)) {
            ++at;
            TranscriptEvent event;
            if (!schema::transcript(line, event)) {
                reason = "Transcript line " + std::to_string(at) + " was refused.";
                agents.clear();
                return false;
            }
            fold_event(made, std::move(event));
        }
        if (!file.eof()) {
            reason = "The transcript file does not read.";
            agents.clear();
            return false;
        }
        agents.push_back(std::move(made));
    }
    if (error) {
        reason = "The transcript directory does not read.";
        return false;
    }
    std::sort(agents.begin(), agents.end(),
              [](const Agent &left, const Agent &right) { return left.id < right.id; });
    reason = "The past transcript is read-only.";
    return true;
}

bool verify_fixtures()
{
    Request request_fixture;
    std::string key;
    std::string value;
    std::string module_name;
    std::string module_kind;
    std::string fetch_name;
    std::string fetch_state;
    std::string loaded;
    std::string phase_word;
    PendingRequest pending_fixture;
    AgentState agent_fixture;
    std::uint64_t fetched = 0u;
    std::uint64_t total = 0u;
    const std::string request_line =
        "{\"request\":41,\"agent\":2,\"turn\":3,\"tool\":\"fs_read\","
        "\"side\":\"host\",\"number\":3,\"arg\":\"path=hello.txt\","
        "\"deadline\":0,\"auth\":\"pending\",\"tick\":7}";
    const bool panel_fixtures =
        schema::request(request_line, request_fixture) &&
        request_fixture.authorization == "pending" &&
        !schema::request(request_line + "x", request_fixture) &&
        schema::setting_result("setting decode.reply_limit 37", key, value) &&
        key == "decode.reply_limit" && value == "37" &&
        !schema::setting_result("setting decode.reply_limit", key, value) &&
        schema::import_result("module check_skill skill import 4 from /tmp/check_skill",
                              module_name, module_kind) &&
        module_name == "check_skill" && module_kind == "skill" &&
        !schema::import_result("module check_skill other import 4 from /tmp/check_skill",
                               module_name, module_kind) &&
        schema::fetch_result("fetch language 7 of 19", fetch_name, fetched, total,
                             fetch_state) &&
        fetch_name == "language" && fetched == 7u && total == 19u &&
        !schema::fetch_result("fetch language 19 of 0", fetch_name, fetched, total,
                              fetch_state) &&
        schema::model_load("model language loaded model-q8.gguf at tick 37", "language",
                           loaded) && loaded == "model-q8.gguf" &&
        !schema::model_load("model language loaded model-q8.gguf at tick x", "language",
                            loaded) &&
        schema::pending_request("request 1042 pending fs_read agent 42 turn 2 path file-42.txt",
                                pending_fixture) &&
        pending_fixture.request == 1042u && pending_fixture.agent == 42u &&
        pending_fixture.turn == 2u && pending_fixture.tool == "fs_read" &&
        pending_fixture.path == "file-42.txt" &&
        !schema::pending_request("request x pending fs_read agent 42 turn 2 path file-42.txt",
                                 pending_fixture) &&
        schema::phase("placing 1", phase_word) && phase_word == "placing" &&
        schema::phase("replaying 2", phase_word) && phase_word == "replaying" &&
        schema::phase("running 3", phase_word) && phase_word == "running" &&
        schema::phase("closed 4", phase_word) && phase_word == "closed" &&
        !schema::phase("ready 5", phase_word) &&
        schema::agent_state("agent 7 turn role 2 parent 0 state 4 turn 3 ticks 91",
                            agent_fixture) &&
        agent_fixture.agent == 7u && agent_fixture.state == 4u && agent_fixture.turn == 3u &&
        !schema::agent_state("agent 7 turn role 2 parent 0 state 9 turn 3 ticks 91",
                             agent_fixture);
    if (!panel_fixtures) return false;
    std::array<char, 40> pattern{};
    const std::string base = "/tmp/aotx_ctrl_replica_XXXXXX";
    std::copy(base.begin(), base.end(), pattern.begin());
    char *made = mkdtemp(pattern.data());
    if (made == nullptr) return false;
    const std::filesystem::path root(made);
    const std::filesystem::path boot = root / "0000000000000001";
    std::error_code error;
    std::filesystem::create_directories(boot / "transcript", error);
    std::filesystem::create_directories(root / "bus", error);
    if (error) return false;
    {
        std::ofstream(boot / "transcript/0.jsonl")
            << "{\"tick\":6,\"kind\":\"part\",\"text\":\"First \","
               "\"request\":0,\"status\":\"open\",\"turn\":1}\n"
            << "{\"tick\":7,\"kind\":\"part\",\"text\":\"second\","
               "\"request\":0,\"status\":\"open\",\"turn\":1}\n";
        std::ofstream(root / "bus/2000-01-01-aotx.jsonl")
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"system\",\"seq\":1,"
               "\"ts\":\"2000-01-01T00:00:00.000+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"sequence done slot 0\",\"tick\":8,"
               "\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n"
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"agent-0\",\"seq\":2,"
               "\"ts\":\"2000-01-01T00:00:00.100+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"request 41 pending fs_read agent 0 turn 1 path hello.txt\","
               "\"tick\":8,\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n"
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"agent-0\",\"seq\":3,"
               "\"ts\":\"2000-01-01T00:00:00.200+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"agent 0 spawned role 1 parent 0 state 1 turn 0 ticks 8\","
               "\"tick\":8,\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n";
        std::ofstream(root / "requests.jsonl")
            << "{\"request\":41,\"agent\":0,\"turn\":1,\"tool\":\"fs_read\","
               "\"side\":\"host\",\"number\":3,\"arg\":\"\\u001fpath=hello.txt\","
               "\"deadline\":500,\"auth\":\"pending\",\"tick\":7}\n";
        std::ofstream(root / "modules.jsonl")
            << "{\"name\":\"reader\",\"kind\":\"tool\",\"side\":\"host\","
               "\"dir\":\"tools/reader\",\"program\":\"run\",\"timeout\":30,"
               "\"authorize\":\"never\",\"import\":1,\"number\":17}\n";
        std::ofstream(root / "phase") << "running 1\n";
        std::ofstream(root / "settings") << "journal.dir = " << root.string() << "\n";
    }
    State state(root, root / "settings");
    std::string configured;
    const bool opened = state.open();
    const bool folded = opened && state.agents().size() == 1u &&
                        state.agents()[0].transcript.size() == 1u &&
                        state.agents()[0].transcript[0].kind == "part" &&
                        state.agents()[0].transcript[0].text == "First second" &&
                        state.agents()[0].part_lines == 2u &&
                        state.pending_requests().size() == 1u &&
                        state.agent_states().size() == 1u &&
                        state.take_results().size() == 1u;
    {
        std::ofstream transcript(boot / "transcript/0.jsonl", std::ios::app);
        transcript << "{\"tick\":8,\"kind\":\"reply\",\"text\":\"First second\","
                      "\"request\":0,\"status\":\"\",\"turn\":1}\n"
                   << "{\"tick\":9,\"kind\":\"bound\",\"text\":\"\","
                      "\"request\":0,\"status\":\"limit\",\"turn\":1}\n"
                   << "{\"tick\":10,\"kind\":\"grant\",\"tool\":\"fs_read\","
                      "\"request\":41,\"status\":\"granted\",\"turn\":1}\n"
                   << "{\"tick\":11,\"kind\":\"unknown\"}\n";
    }
    {
        std::ofstream(root / "bus/2000-01-01-aotx.jsonl", std::ios::app)
            << "{\"v\":1,\"run\":\"aotx\",\"agent\":\"system\",\"seq\":2,"
               "\"ts\":\"2000-01-01T00:00:01.000+00:00\",\"type\":\"note\","
               "\"body\":{\"text\":\"model language loaded model-q8.gguf at tick 37\","
               "\"tick\":9,\"boot\":\"0000000000000001\",\"lag_ms\":null}}\n";
    }
    state.tick(1.0);
    const std::vector<std::string> final_results = state.take_results();
    const bool valid = folded && setting_value(root / "settings", "journal.dir", configured) &&
                       configured == root.string() && state.phase() == "running" &&
                       state.agents().size() == 1u &&
                       state.agents()[0].transcript.size() == 3u &&
                       state.agents()[0].transcript[0].kind == "reply" &&
                       state.agents()[0].transcript[0].text == "First second" &&
                       state.agents()[0].transcript[1].kind == "bound" &&
                       state.agents()[0].transcript[1].status == "limit" &&
                       state.agents()[0].transcript[2].kind == "grant" &&
                       state.agents()[0].fold_replaced && state.agents()[0].reply_bound &&
                       state.language_model() == "model-q8.gguf" &&
                       state.notes().size() == 4u && state.requests().size() == 1u &&
                       state.requests()[0].argument == "\x1fpath=hello.txt" &&
                       state.pending_requests().empty() &&
                       state.modules().size() == 1u && final_results.size() == 2u &&
                       std::find(final_results.begin(), final_results.end(),
                                 "model language loaded model-q8.gguf at tick 37") !=
                           final_results.end();
    std::filesystem::remove_all(root, error);
    return valid;
}

} // namespace aotx::ctrl::replica
