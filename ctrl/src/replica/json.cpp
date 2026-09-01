// Purpose: Parse the JSON object lines of replica files.
// Owns: No state outside one parser call.
// Launch shape: One caller parses one complete line.
// Lifetime: Temporary parser state ends when the call returns.
#include "replica/json.hpp"

#include <charconv>
#include <cstdint>
#include <limits>
#include <string_view>

namespace aotx::ctrl::replica::json {
namespace {

void append_utf8(std::string &out, unsigned value)
{
    if (value <= 0x7fu) {
        out.push_back(static_cast<char>(value));
    } else if (value <= 0x7ffu) {
        out.push_back(static_cast<char>(0xc0u | (value >> 6u)));
        out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
    } else {
        out.push_back(static_cast<char>(0xe0u | (value >> 12u)));
        out.push_back(static_cast<char>(0x80u | ((value >> 6u) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (value & 0x3fu)));
    }
}

class Parser {
  public:
    explicit Parser(std::string_view source) : source_(source) {}

    bool run(Value &out)
    {
        space();
        if (!value(out)) return false;
        space();
        return at_ == source_.size();
    }

  private:
    void space()
    {
        while (at_ < source_.size() &&
               (source_[at_] == ' ' || source_[at_] == '\t' ||
                source_[at_] == '\r' || source_[at_] == '\n')) ++at_;
    }

    bool take(char byte)
    {
        if (at_ >= source_.size() || source_[at_] != byte) return false;
        ++at_;
        return true;
    }

    bool literal(std::string_view word)
    {
        if (source_.substr(at_, word.size()) != word) return false;
        at_ += word.size();
        return true;
    }

    bool hex(unsigned &value)
    {
        value = 0u;
        for (unsigned index = 0u; index < 4u; ++index) {
            if (at_ >= source_.size()) return false;
            const char byte = source_[at_++];
            unsigned digit;
            if (byte >= '0' && byte <= '9') digit = static_cast<unsigned>(byte - '0');
            else if (byte >= 'a' && byte <= 'f') digit = 10u + static_cast<unsigned>(byte - 'a');
            else if (byte >= 'A' && byte <= 'F') digit = 10u + static_cast<unsigned>(byte - 'A');
            else return false;
            value = value * 16u + digit;
        }
        return true;
    }

    bool string(std::string &out)
    {
        if (!take('"')) return false;
        out.clear();
        while (at_ < source_.size()) {
            const unsigned char byte = static_cast<unsigned char>(source_[at_++]);
            if (byte == '"') return true;
            if (byte < 0x20u) return false;
            if (byte != '\\') {
                out.push_back(static_cast<char>(byte));
                continue;
            }
            if (at_ >= source_.size()) return false;
            const char escaped = source_[at_++];
            if (escaped == '"' || escaped == '\\' || escaped == '/') out.push_back(escaped);
            else if (escaped == 'b') out.push_back('\b');
            else if (escaped == 'f') out.push_back('\f');
            else if (escaped == 'n') out.push_back('\n');
            else if (escaped == 'r') out.push_back('\r');
            else if (escaped == 't') out.push_back('\t');
            else if (escaped == 'u') {
                unsigned code;
                if (!hex(code) || (code >= 0xd800u && code <= 0xdfffu)) return false;
                append_utf8(out, code);
            } else return false;
        }
        return false;
    }

    bool number_value(Value &out)
    {
        const std::size_t start = at_;
        if (at_ < source_.size() && source_[at_] == '-') ++at_;
        if (at_ >= source_.size()) return false;
        if (source_[at_] == '0') {
            ++at_;
        } else {
            if (source_[at_] < '1' || source_[at_] > '9') return false;
            while (at_ < source_.size() && source_[at_] >= '0' && source_[at_] <= '9') ++at_;
        }
        if (at_ < source_.size() && source_[at_] == '.') {
            ++at_;
            const std::size_t digits = at_;
            while (at_ < source_.size() && source_[at_] >= '0' && source_[at_] <= '9') ++at_;
            if (digits == at_) return false;
        }
        if (at_ < source_.size() && (source_[at_] == 'e' || source_[at_] == 'E')) {
            ++at_;
            if (at_ < source_.size() && (source_[at_] == '+' || source_[at_] == '-')) ++at_;
            const std::size_t digits = at_;
            while (at_ < source_.size() && source_[at_] >= '0' && source_[at_] <= '9') ++at_;
            if (digits == at_) return false;
        }
        out.kind = Kind::number;
        out.text.assign(source_.substr(start, at_ - start));
        return true;
    }

    bool object(Value &out)
    {
        if (!take('{')) return false;
        out.kind = Kind::object;
        out.members.clear();
        space();
        if (take('}')) return true;
        while (true) {
            std::string name;
            Value child;
            if (!string(name)) return false;
            for (const auto &held : out.members) if (held.first == name) return false;
            space();
            if (!take(':')) return false;
            space();
            if (!value(child)) return false;
            out.members.emplace_back(std::move(name), std::move(child));
            space();
            if (take('}')) return true;
            if (!take(',')) return false;
            space();
        }
    }

    bool value(Value &out)
    {
        if (at_ >= source_.size()) return false;
        if (source_[at_] == '{') return object(out);
        if (source_[at_] == '"') {
            out.kind = Kind::string;
            return string(out.text);
        }
        if (source_[at_] == '-' || (source_[at_] >= '0' && source_[at_] <= '9')) {
            return number_value(out);
        }
        if (literal("null")) {
            out.kind = Kind::null_value;
            return true;
        }
        if (literal("true")) {
            out.kind = Kind::boolean;
            out.boolean = true;
            return true;
        }
        if (literal("false")) {
            out.kind = Kind::boolean;
            out.boolean = false;
            return true;
        }
        return false;
    }

    std::string_view source_;
    std::size_t at_ = 0u;
};

} // namespace

const Value *Value::get(const char *name) const
{
    if (kind != Kind::object) return nullptr;
    for (const auto &member : members) if (member.first == name) return &member.second;
    return nullptr;
}

bool parse(const std::string &line, Value &out)
{
    return Parser(line).run(out);
}

bool text(const Value &object, const char *name, std::string &out)
{
    const Value *value = object.get(name);
    if (value == nullptr || value->kind != Kind::string) return false;
    out = value->text;
    return true;
}

bool number(const Value &object, const char *name, unsigned long long &out)
{
    const Value *value = object.get(name);
    if (value == nullptr || value->kind != Kind::number || value->text.empty() ||
        value->text.front() == '-' || value->text.find_first_of(".eE") != std::string::npos) {
        return false;
    }
    const auto parsed = std::from_chars(value->text.data(), value->text.data() + value->text.size(), out);
    return parsed.ec == std::errc() && parsed.ptr == value->text.data() + value->text.size();
}

} // namespace aotx::ctrl::replica::json
