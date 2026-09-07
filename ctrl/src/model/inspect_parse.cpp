// Purpose: Validate the complete report from the model header inspector.
// Owns: Report syntax checks and parsed header facts.
// Threading: One interface thread parses a bounded report after pipe closure.
// Lifetime: No state remains after the call.
#include "model/inspect_parse.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <set>
#include <string_view>

namespace aotx::ctrl::model {
namespace {

using Text = std::string_view;

struct Report {
    Text text;
    Text line;

    bool next()
    {
        const auto end = text.find('\n');
        if (end == Text::npos) return false;
        line = text.substr(0, end);
        text.remove_prefix(end + 1);
        return true;
    }

    bool field(Text prefix, Text &value)
    {
        if (!next() || line.substr(0, prefix.size()) != prefix) return false;
        value = line.substr(prefix.size());
        return true;
    }

    bool exact(Text value) { return next() && line == value; }
};

bool take(Text &value, Text suffix, Text &head)
{
    const auto at = value.find(suffix);
    if (at == Text::npos) return false;
    head = value.substr(0, at);
    value.remove_prefix(at + suffix.size());
    return true;
}

template<class T> bool number(Text value, T &out)
{
    if (value.empty() || value.front() < '0' || value.front() > '9') return false;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), out);
    return result.ec == std::errc{} && result.ptr == value.data() + value.size();
}

bool yes_no(Text value, bool &out)
{
    if (value != "yes" && value != "no") return false;
    out = value == "yes";
    return true;
}

bool text_value(Text value, std::string &out)
{
    for (std::size_t i = 0; i < value.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        if (c <= 32 || c >= 127) return false;
        if (c == '\\') {
            if (i + 3 >= value.size() || value[i + 1] != 'x') return false;
            for (std::size_t j = i + 2; j <= i + 3; ++j)
                if (!((value[j] >= '0' && value[j] <= '9') ||
                      (value[j] >= 'a' && value[j] <= 'f'))) return false;
            i += 3;
        }
    }
    out = value;
    return true;
}

std::string escaped(const std::string &source)
{
    static const char hex[] = "0123456789abcdef";
    std::string result;
    const auto limit = std::min(source.size(), std::size_t{512});
    for (std::size_t i = 0; i < limit; ++i) {
        const unsigned char c = static_cast<unsigned char>(source[i]);
        if (c > 32 && c < 127 && c != '\\') result += static_cast<char>(c);
        else {
            result += "\\x";
            result += hex[c >> 4];
            result += hex[c & 15];
        }
    }
    if (source.size() > limit) result += "[cut]";
    return result;
}

bool named_support(Report &report, Text prefix, std::string &name, bool &supported)
{
    Text value, head;
    return report.field(prefix, value) && take(value, " supported=", head) &&
           text_value(head, name) && yes_no(value, supported);
}

bool block(Text value, InspectBlockType &out)
{
    Text name, id, count;
    return take(value, " id=", name) && take(value, " count=", id) &&
           take(value, " supported=", count) && !name.empty() &&
           text_value(name, out.name) && number(id, out.id) &&
           number(count, out.count) && out.count != 0 && yes_no(value, out.supported);
}

bool layer(Text value, InspectLayerType &out)
{
    Text name;
    return take(value, " count=", name) && !name.empty() && text_value(name, out.name) &&
           number(value, out.count) && out.count != 0;
}

bool digest(Text value, std::string &out)
{
    if (value.size() != 64) return false;
    for (char c : value)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return false;
    out = value;
    return true;
}

} // namespace

std::optional<InspectHeader> aotx_parse_inspection(const std::string &output,
                                                 const std::string &source)
{
    InspectHeader out;
    Report report{output, {}};
    Text value, first, second;
    if (!report.field("file=", value) || value != escaped(source)) return {};
    out.file = value;
    if (!report.field("architecture=", value) || !text_value(value, out.architecture) ||
        !named_support(report, "rotary_pairs=", out.rotary_pairs, out.rotary_pairs_supported) ||
        !named_support(report, "pre_tokenizer=", out.pre_tokenizer, out.pre_tokenizer_supported) ||
        !named_support(report, "tokenizer_model=", out.tokenizer_model, out.tokenizer_model_supported) ||
        !report.field("tensors=", value) || !number(value, out.tensors)) return {};
    std::set<std::uint32_t> ids;
    std::uint64_t total = 0;
    while (report.text.substr(0, 11) == "block_type=") {
        InspectBlockType item;
        if (!report.field("block_type=", value) || !block(value, item) ||
            !ids.insert(item.id).second || item.count > out.tensors - total) return {};
        total += item.count;
        out.block_types.push_back(std::move(item));
    }
    if (total != out.tensors || !report.field("layers=", value) ||
        !take(value, " hidden=", first) || !take(value, " vocabulary=", second) ||
        !number(first, out.layers) || !number(second, out.hidden) ||
        !number(value, out.vocabulary)) return {};
    std::set<std::string> names;
    total = 0;
    while (report.text.substr(0, 11) == "layer_type=") {
        InspectLayerType item;
        if (!report.field("layer_type=", value) || !layer(value, item) ||
            !names.insert(item.name).second || item.count > out.layers - total) return {};
        total += item.count;
        out.layer_types.push_back(std::move(item));
    }
    if (!report.field("layer_sets_supported=", value) ||
        !take(value, " unknown_tensors=", first) || !take(value, " layer_limit=", second) ||
        !yes_no(first, out.layer_sets_supported) || !number(second, out.unknown_tensors) ||
        (out.layer_sets_supported && total != out.layers) ||
        !number(value, out.layer_limit) || !report.field("chat_template_bytes=", value) ||
        !take(value, " chat_template_sha256=", first) ||
        !number(first, out.chat_template_bytes) || !digest(value, out.chat_template_sha256) ||
        !report.field("file_bytes=", value) || !number(value, out.file_bytes) ||
        !report.field("header_bytes=", value) || !number(value, out.header_bytes) ||
        out.header_bytes > out.file_bytes) return {};
    if (source.compare(0, 7, "http://") == 0 || source.compare(0, 8, "https://") == 0) {
        std::uint64_t received = 0;
        if (!report.field("received_bytes=", value) || !number(value, received)) return {};
        out.received_bytes = received;
    }
    if (!report.field("build_support=", value) || !yes_no(value, out.build_support) ||
        !report.exact("run_verified=no") ||
        !report.exact("The support result covers the listed header fields and tensor sets only.") ||
        !report.exact("The header does not prove weight integrity, memory fit, wrap, prefill, or restore."))
        return {};
    if (!out.build_support && !report.exact(
        "This build cannot run this file with the listed unsupported or missing fields.")) return {};
    if (!report.text.empty()) return {};
    return out;
}

} // namespace aotx::ctrl::model
