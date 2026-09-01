// Purpose: Define the bounded reply formatting document.
// Owns: No state; callers own all parsed blocks and spans.
// Launch shape: One pure parse call for each reply text.
// Lifetime: A document remains valid while its caller keeps it.
#ifndef AOTX_CTRL_CHAT_FORMAT_HPP
#define AOTX_CTRL_CHAT_FORMAT_HPP

#include <string>
#include <vector>

namespace aotx::ctrl::chat::format {

enum class SpanKind { text, bold, italic, code };
enum class BlockKind { paragraph, heading, bullet, number, code };

struct Span {
    SpanKind kind = SpanKind::text;
    std::string text;
};

struct Block {
    BlockKind kind = BlockKind::paragraph;
    unsigned level = 0u;
    std::vector<Span> spans;
    std::string text;
};

using Document = std::vector<Block>;

Document render(const std::string &reply);
std::size_t word_end(const std::string &text, std::size_t at);
bool verify_fixtures();

} // namespace aotx::ctrl::chat::format

#endif
