// Purpose: Parse transcript, note, request, module, and model lines.
// Owns: No state outside one parser call.
// Launch shape: One call validates one complete JSON object.
// Lifetime: Temporary JSON values end when the call returns.
#include "replica/schema.hpp"

#include "replica/json.hpp"

#include <array>
#include <charconv>
#include <string>
#include <system_error>

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
        !json::text(value, "auth", made.authorization) || made.authorization.empty() ||
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

} // namespace aotx::ctrl::replica::schema
