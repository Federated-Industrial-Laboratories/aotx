// Purpose: Parse the bounded reply formatting grammar.
// Owns: Temporary line and span state for one pure parse call.
// Launch shape: One bounded forward pass over each reply.
// Lifetime: Temporary parser state ends when the call returns.
#include "chat/format.hpp"

#include <cctype>
#include <string_view>

namespace aotx::ctrl::chat::format {
namespace {

void plain(std::vector<Span> &out, std::string text)
{
    if (text.empty()) return;
    if (!out.empty() && out.back().kind == SpanKind::text) out.back().text += text;
    else out.push_back({SpanKind::text, std::move(text)});
}

std::vector<Span> inline_spans(std::string_view line)
{
    std::vector<Span> out;
    std::size_t at = 0u;
    while (at < line.size()) {
        SpanKind kind = SpanKind::text;
        std::string_view mark;
        if (line.substr(at, 2u) == "**") {
            kind = SpanKind::bold;
            mark = "**";
        } else if (line[at] == '*') {
            kind = SpanKind::italic;
            mark = "*";
        } else if (line[at] == '`') {
            kind = SpanKind::code;
            mark = "`";
        } else {
            const std::size_t next = line.find_first_of("*`", at);
            plain(out, std::string(line.substr(at, next - at)));
            at = next == std::string_view::npos ? line.size() : next;
            continue;
        }
        const std::size_t close = line.find(mark, at + mark.size());
        if (close == std::string_view::npos || close == at + mark.size()) {
            plain(out, std::string(line.substr(at)));
            break;
        }
        out.push_back({kind, std::string(line.substr(at + mark.size(),
                                                     close - at - mark.size()))});
        at = close + mark.size();
    }
    return out;
}

bool numbered(std::string_view line, std::size_t &text_at, unsigned &number)
{
    std::size_t at = 0u;
    unsigned value = 0u;
    while (at < line.size() && std::isdigit(static_cast<unsigned char>(line[at]))) {
        if (value > 100000u) return false;
        value = value * 10u + static_cast<unsigned>(line[at] - '0');
        ++at;
    }
    if (at == 0u || at + 1u >= line.size() || line[at] != '.' || line[at + 1u] != ' ') {
        return false;
    }
    text_at = at + 2u;
    number = value;
    return true;
}

Block line_block(std::string_view line)
{
    Block block;
    std::size_t heading = 0u;
    while (heading < line.size() && heading < 6u && line[heading] == '#') ++heading;
    std::size_t text_at = 0u;
    unsigned number = 0u;
    if (heading > 0u && heading < line.size() && line[heading] == ' ') {
        block.kind = BlockKind::heading;
        block.level = static_cast<unsigned>(heading);
        text_at = heading + 1u;
    } else if (line.size() > 2u && (line.substr(0u, 2u) == "- " ||
                                   line.substr(0u, 2u) == "* ")) {
        block.kind = BlockKind::bullet;
        text_at = 2u;
    } else if (numbered(line, text_at, number)) {
        block.kind = BlockKind::number;
        block.level = number;
    }
    block.spans = inline_spans(line.substr(text_at));
    return block;
}

bool same_span(const Span &left, SpanKind kind, const char *text)
{
    return left.kind == kind && left.text == text;
}

} // namespace

Document render(const std::string &reply)
{
    Document out;
    std::size_t at = 0u;
    while (at < reply.size()) {
        const std::size_t end = reply.find('\n', at);
        const std::size_t count = end == std::string::npos ? reply.size() - at : end - at;
        const std::string_view line(reply.data() + at, count);
        if (line.substr(0u, 3u) == "```") {
            const std::size_t body = end == std::string::npos ? reply.size() : end + 1u;
            std::size_t scan = body;
            std::size_t close = std::string::npos;
            while (scan < reply.size()) {
                const std::size_t close_end = reply.find('\n', scan);
                const std::size_t close_count = close_end == std::string::npos
                    ? reply.size() - scan : close_end - scan;
                if (std::string_view(reply.data() + scan, close_count) == "```") {
                    close = scan;
                    at = close_end == std::string::npos ? reply.size() : close_end + 1u;
                    break;
                }
                scan = close_end == std::string::npos ? reply.size() : close_end + 1u;
            }
            if (close != std::string::npos) {
                Block block;
                block.kind = BlockKind::code;
                block.text = reply.substr(body, close - body);
                if (!block.text.empty() && block.text.back() == '\n') block.text.pop_back();
                out.push_back(std::move(block));
                continue;
            }
            Block block;
            block.spans.push_back({SpanKind::text, reply.substr(at)});
            out.push_back(std::move(block));
            break;
        }
        if (!line.empty()) out.push_back(line_block(line));
        else out.push_back(Block{});
        at = end == std::string::npos ? reply.size() : end + 1u;
    }
    return out;
}

bool verify_fixtures()
{
    const Document inline_doc = render("Plain **bold** *soft* `code`");
    const bool inline_ok = inline_doc.size() == 1u && inline_doc[0].spans.size() == 6u &&
        same_span(inline_doc[0].spans[1], SpanKind::bold, "bold") &&
        same_span(inline_doc[0].spans[3], SpanKind::italic, "soft") &&
        same_span(inline_doc[0].spans[5], SpanKind::code, "code");
    const Document blocks = render("## Head\n- one\n7. two\n```cpp\nx();\n```");
    const bool block_ok = blocks.size() == 4u && blocks[0].kind == BlockKind::heading &&
        blocks[0].level == 2u && blocks[1].kind == BlockKind::bullet &&
        blocks[2].kind == BlockKind::number && blocks[2].level == 7u &&
        blocks[3].kind == BlockKind::code && blocks[3].text == "x();";
    const Document malformed = render("**open\n```\nopen");
    const bool malformed_ok = malformed.size() == 2u && malformed[0].spans.size() == 1u &&
        same_span(malformed[0].spans[0], SpanKind::text, "**open") &&
        malformed[1].spans.size() == 1u && malformed[1].spans[0].text == "```\nopen";

    /* Each changed fixture must be rejected by the same structural assertions. */
    const Document inline_mutation = render("Plain **bold* *soft* `code`");
    const Document block_mutation = render("##Head\n-one\n7 two\n```cpp\nx();");
    const Document malformed_mutation = render("**open**\n```\nopen\n```");
    const bool mutations_caught = !(inline_mutation.size() == 1u &&
        inline_mutation[0].spans.size() == 6u) &&
        !(block_mutation.size() == 4u && block_mutation[0].kind == BlockKind::heading &&
          block_mutation[1].kind == BlockKind::bullet &&
          block_mutation[2].kind == BlockKind::number &&
          block_mutation[3].kind == BlockKind::code) &&
        !(malformed_mutation.size() == 2u && malformed_mutation[0].spans.size() == 1u &&
          malformed_mutation[1].spans.size() == 1u);
    return inline_ok && block_ok && malformed_ok && mutations_caught;
}

} // namespace aotx::ctrl::chat::format
