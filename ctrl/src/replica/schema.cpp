// Purpose: Parse transcript, note, request, module, and model lines.
// Owns: No state outside one parser call.
// Launch shape: One call validates one complete JSON object.
// Lifetime: Temporary JSON values end when the call returns.
#include "replica/schema.hpp"

#include "replica/json.hpp"

#include <array>
#include <charconv>
#include <cstdio>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace aotx::ctrl::replica::schema {
namespace {

bool known_kind(const std::string &kind)
{
    static const std::array<const char *, 12> kinds = {
        "line", "part", "bound", "reply", "call", "result", "grant", "refuse",
        "verdict", "done", "selection", "summary"};
    for (const char *held : kinds) if (kind == held) return true;
    return false;
}

bool object(const std::string &line, json::Value &value)
{
    return json::parse(line, value) && value.kind == json::Kind::object;
}

bool exact_text(const json::Value &value, const char *key, const char *wanted)
{
    std::string held;
    return json::text(value, key, held) && held == wanted;
}

bool digest(const std::string &text)
{
    if (text.size() != 64u) return false;
    for (const char byte : text) {
        if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) return false;
    }
    return true;
}

bool model_base(const json::Value &value, Model &made, const char *file_key)
{
    unsigned long long bytes;
    return json::text(value, "name", made.name) && !made.name.empty() &&
           json::text(value, file_key, made.file) && !made.file.empty() &&
           json::number(value, "bytes", bytes) && bytes != 0u &&
           json::text(value, "sha256", made.digest) && digest(made.digest) &&
           ((made.bytes = bytes), true);
}

bool word_tail(const std::string &text, const char *prefix, std::string &word,
               std::string &tail)
{
    const std::size_t start = std::char_traits<char>::length(prefix);
    if (text.rfind(prefix, 0u) != 0u) return false;
    const std::size_t space = text.find(' ', start);
    if (space == std::string::npos || space == start || space + 1u == text.size()) return false;
    word = text.substr(start, space - start);
    tail = text.substr(space + 1u);
    return true;
}

} // namespace

bool transcript(const std::string &line, TranscriptEvent &out)
{
    json::Value value;
    unsigned long long tick;
    unsigned long long request;
    unsigned long long turn;
    TranscriptEvent made;
    if (!object(line, value) || !json::number(value, "tick", tick) ||
        !json::text(value, "kind", made.kind) || !known_kind(made.kind) ||
        !json::number(value, "request", request) ||
        !json::text(value, "status", made.status) ||
        !json::number(value, "turn", turn)) return false;
    const json::Value *text_value = value.get("text");
    const json::Value *tool_value = value.get("tool");
    if (text_value == nullptr && tool_value == nullptr) return false;
    if (text_value != nullptr) {
        if (text_value->kind != json::Kind::string) return false;
        made.text = text_value->text;
    }
    if (tool_value != nullptr) {
        if (tool_value->kind != json::Kind::string || tool_value->text.empty()) return false;
        made.tool = tool_value->text;
    }
    made.tick = tick;
    made.request = request;
    made.turn = turn;
    out = std::move(made);
    return true;
}

bool note(const std::string &line, Note &out)
{
    json::Value value;
    unsigned long long version;
    unsigned long long sequence;
    unsigned long long tick;
    std::string run;
    std::string timestamp;
    Note made;
    if (!object(line, value) || !json::number(value, "v", version) || version != 1u ||
        !json::text(value, "run", run) || run.empty() ||
        !json::text(value, "agent", made.agent) || made.agent.empty() ||
        !json::number(value, "seq", sequence) || sequence == 0u ||
        !json::text(value, "ts", timestamp) || timestamp.empty() ||
        !exact_text(value, "type", "note")) return false;
    const json::Value *body = value.get("body");
    if (body == nullptr || body->kind != json::Kind::object ||
        !json::text(*body, "text", made.text) || made.text.empty() ||
        !json::number(*body, "tick", tick) || !json::text(*body, "boot", made.boot)) {
        return false;
    }
    const json::Value *lag = body->get("lag_ms");
    if (lag == nullptr || (lag->kind != json::Kind::null_value &&
                           lag->kind != json::Kind::number)) return false;
    made.sequence = sequence;
    made.tick = tick;
    out = std::move(made);
    return true;
}

bool request(const std::string &line, Request &out)
{
    json::Value value;
    unsigned long long id;
    unsigned long long agent;
    unsigned long long turn;
    unsigned long long number;
    unsigned long long deadline;
    unsigned long long tick;
    Request made;
    if (!object(line, value) || !json::number(value, "request", id) || id == 0u ||
        !json::number(value, "agent", agent) || !json::number(value, "turn", turn) ||
        !json::text(value, "tool", made.tool) || made.tool.empty() ||
        !json::text(value, "side", made.side) || made.side.empty() ||
        !json::number(value, "number", number) ||
        !json::text(value, "arg", made.argument) ||
        !json::number(value, "deadline", deadline) ||
        !json::text(value, "auth", made.authorization) ||
        (made.authorization != "none" && made.authorization != "pending" &&
         made.authorization != "granted" && made.authorization != "refused") ||
        !json::number(value, "tick", tick)) return false;
    made.request = id;
    made.agent = agent;
    made.turn = turn;
    made.number = number;
    made.deadline = deadline;
    made.tick = tick;
    out = std::move(made);
    return true;
}

bool pending_request(const std::string &text, PendingRequest &out)
{
    PendingRequest made;
    std::string request_text;
    std::string tail;
    if (!word_tail(text, "request ", request_text, tail)) return false;
    const auto request = std::from_chars(request_text.data(),
                                         request_text.data() + request_text.size(),
                                         made.request);
    if (request.ec != std::errc() || request.ptr != request_text.data() + request_text.size() ||
        made.request == 0u || tail.rfind("pending ", 0u) != 0u) return false;
    tail.erase(0u, 8u);
    const std::size_t agent_at = tail.find(" agent ");
    const std::size_t turn_at = tail.find(" turn ", agent_at == std::string::npos ? 0u : agent_at);
    const std::size_t path_at = tail.find(" path ", turn_at == std::string::npos ? 0u : turn_at);
    if (agent_at == std::string::npos || turn_at == std::string::npos ||
        path_at == std::string::npos || agent_at == 0u || path_at + 6u > tail.size()) return false;
    made.tool = tail.substr(0u, agent_at);
    const auto agent = std::from_chars(tail.data() + agent_at + 7u, tail.data() + turn_at,
                                       made.agent);
    const auto turn = std::from_chars(tail.data() + turn_at + 6u, tail.data() + path_at,
                                      made.turn);
    if (agent.ec != std::errc() || agent.ptr != tail.data() + turn_at ||
        turn.ec != std::errc() || turn.ptr != tail.data() + path_at) return false;
    made.path = tail.substr(path_at + 6u);
    out = std::move(made);
    return true;
}

bool agent_state(const std::string &text, AgentState &out)
{
    AgentState made;
    char event[16]{};
    unsigned long long agent = 0u, role = 0u, parent = 0u, state = 0u, turn = 0u, ticks = 0u;
    char extra = '\0';
    const int fields = std::sscanf(text.c_str(),
        "agent %llu %15s role %llu parent %llu state %llu turn %llu ticks %llu %c",
        &agent, event, &role, &parent, &state, &turn, &ticks, &extra);
    if (fields != 7 || (std::string(event) != "spawned" && std::string(event) != "turn" &&
                        std::string(event) != "released") || state > 5u) return false;
    made.agent = agent;
    made.event = event;
    made.role = role;
    made.parent = parent;
    made.state = state;
    made.turn = turn;
    made.ticks = ticks;
    out = std::move(made);
    return true;
}

bool phase(const std::string &line, std::string &word)
{
    std::istringstream input(line);
    long long seconds = 0;
    std::string extra;
    if (!(input >> word >> seconds) || (input >> extra) || seconds < 0 ||
        (word != "placing" && word != "replaying" && word != "running" &&
         word != "closed")) return false;
    return true;
}

bool module(const std::string &line, Module &out)
{
    json::Value value;
    unsigned long long timeout;
    unsigned long long import;
    unsigned long long number;
    Module made;
    if (!object(line, value) || !json::text(value, "name", made.name) || made.name.empty() ||
        !json::text(value, "kind", made.kind) || made.kind.empty() ||
        !json::text(value, "side", made.side) || made.side.empty() ||
        !json::text(value, "dir", made.directory) ||
        !json::text(value, "program", made.program) ||
        !json::number(value, "timeout", timeout) ||
        !json::text(value, "authorize", made.authorization) ||
        !json::number(value, "import", import) ||
        !json::number(value, "number", number)) return false;
    made.timeout = timeout;
    made.import = import;
    made.number = number;
    out = std::move(made);
    return true;
}

bool model_catalog(const std::string &line, Model &out)
{
    json::Value value;
    Model made;
    std::string repository;
    std::string revision;
    std::string license;
    std::string profiles;
    std::string note;
    if (!object(line, value) || !model_base(value, made, "file") ||
        !json::text(value, "role", made.role) || made.role.empty() ||
        !json::text(value, "repository", repository) || repository.empty() ||
        !json::text(value, "revision", revision) || revision.empty() ||
        !json::text(value, "license", license) || license.empty() ||
        !json::text(value, "quant", made.quant) || made.quant.empty() ||
        !json::text(value, "profiles", profiles) || profiles.empty() ||
        !json::text(value, "source", made.source) || made.source.empty() ||
        !json::text(value, "note", note)) return false;
    const json::Value *verified = value.get("verified");
    if (verified == nullptr || verified->kind != json::Kind::boolean) return false;
    made.verified = verified->boolean;
    out = std::move(made);
    return true;
}

bool model_store(const std::string &line, Model &out)
{
    json::Value value;
    Model made;
    std::string date;
    std::string revision;
    if (!object(line, value) || !model_base(value, made, "file") ||
        !json::text(value, "source", made.source) || made.source.empty() ||
        !json::text(value, "date", date) || date.empty() ||
        !json::text(value, "revision", revision) || revision.empty()) return false;
    const json::Value *verified = value.get("verified");
    if (verified == nullptr || verified->kind != json::Kind::boolean || !verified->boolean) {
        return false;
    }
    made.verified = true;
    out = std::move(made);
    return true;
}

bool model_manifest(const std::string &line, Model &out)
{
    json::Value value;
    Model made;
    std::string revision;
    std::string license;
    if (!object(line, value) || !model_base(value, made, "path") ||
        !json::text(value, "source", made.source) ||
        !json::text(value, "revision", revision) ||
        !json::text(value, "license", license)) return false;
    if (!json::text(value, "role", made.role)) made.role = made.name;
    if (made.role.empty()) return false;
    made.active = true;
    out = std::move(made);
    return true;
}

bool language_model(const std::string &line, const std::string &wanted, std::string &name)
{
    json::Value value;
    unsigned long long bytes;
    std::string role;
    std::string path;
    std::string source;
    std::string revision;
    std::string license;
    std::string digest;
    std::string held;
    if (!object(line, value) || !json::text(value, "name", held) || held.empty() ||
        !json::text(value, "path", path) || path.empty() ||
        !json::text(value, "source", source) || !json::text(value, "revision", revision) ||
        !json::text(value, "license", license) || !json::number(value, "bytes", bytes) ||
        !json::text(value, "sha256", digest) || digest.size() != 64u) return false;
    if (!json::text(value, "role", role)) role = held;
    for (const char byte : digest) {
        if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f'))) return false;
    }
    if (role != wanted) return true;
    name = held;
    return true;
}

bool model_load(const std::string &text, const std::string &wanted, std::string &file)
{
    constexpr const char *prefix = "model ";
    constexpr const char *loaded = " loaded ";
    constexpr const char *at_tick = " at tick ";
    if (text.rfind(prefix, 0u) != 0u) return false;
    const std::size_t role_end = text.find(loaded, 6u);
    if (role_end == std::string::npos || role_end == 6u) return false;
    const std::size_t file_start = role_end + 8u;
    const std::size_t file_end = text.rfind(at_tick);
    if (file_end == std::string::npos || file_end <= file_start) return false;
    const std::size_t tick_start = file_end + 9u;
    unsigned long long tick = 0u;
    const auto parsed = std::from_chars(text.data() + tick_start,
                                        text.data() + text.size(), tick);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size()) return false;
    if (text.substr(6u, role_end - 6u) == wanted) {
        file = text.substr(file_start, file_end - file_start);
    }
    return true;
}

bool setting_result(const std::string &text, std::string &key, std::string &value)
{
    std::string tail;
    return word_tail(text, "setting ", key, value) && value.find(' ') == std::string::npos;
}

bool import_result(const std::string &text, std::string &name, std::string &kind)
{
    std::string tail;
    if (!word_tail(text, "module ", name, tail)) return false;
    const std::size_t mark = tail.find(" import ");
    if (mark == std::string::npos || mark == 0u || tail.find(" from ", mark + 8u) ==
        std::string::npos) return false;
    kind = tail.substr(0u, mark);
    return kind == "skill" || kind == "role" || kind == "tool";
}

bool fetch_result(const std::string &text, std::string &name, std::uint64_t &done,
                  std::uint64_t &total, std::string &result)
{
    std::string tail;
    if (!word_tail(text, "fetch ", name, tail)) return false;
    const std::size_t of = tail.find(" of ");
    if (of != std::string::npos) {
        const auto left = std::from_chars(tail.data(), tail.data() + of, done);
        const auto right = std::from_chars(tail.data() + of + 4u,
                                           tail.data() + tail.size(), total);
        if (left.ec == std::errc() && left.ptr == tail.data() + of &&
            right.ec == std::errc() && right.ptr == tail.data() + tail.size() && total != 0u) {
            result = "progress";
            return true;
        }
    }
    result = tail;
    return tail == "started" || tail == "on disk" || tail == "failed" ||
           tail.rfind("host ", 0u) == 0u || tail.rfind("restart ", 0u) == 0u ||
           tail.rfind("refused because ", 0u) == 0u;
}

bool action_result(const std::string &text)
{
    std::string first;
    std::string second;
    std::uint64_t done = 0u;
    std::uint64_t total = 0u;
    std::string result;
    if (setting_result(text, first, second) || import_result(text, first, second) ||
        fetch_result(text, first, done, total, result) ||
        model_load(text, "language", second) || model_load(text, "language-q4", second) ||
        model_load(text, "embedding", second) || model_load(text, "reranker", second)) {
        return true;
    }
    if (text.rfind("set:", 0u) == 0u) {
        static const std::array<const char *, 5> effects = {
            " tick", " sequence", " task", " request", " frame"};
        for (const char *effect : effects) {
            const std::size_t size = std::char_traits<char>::length(effect);
            if (text.size() >= size && text.compare(text.size() - size, size, effect) == 0) {
                return false;
            }
        }
        return true;
    }
    if (text.rfind("model load:", 0u) == 0u) {
        return text.find(" takes ") == std::string::npos;
    }
    static const std::array<const char *, 7> prefixes = {
        "authorize:", "refuse:", "agent:", "model fetch:", "import ", "module ",
        "continue:"};
    for (const char *prefix : prefixes) if (text.rfind(prefix, 0u) == 0u) return true;
    return false;
}

} // namespace aotx::ctrl::replica::schema
