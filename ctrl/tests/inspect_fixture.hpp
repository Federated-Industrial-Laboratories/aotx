// Purpose: Supply distinct local reports for the inspection child checks.
// Owns: Test report bytes and child failure modes.
// Threading: One test child writes one report to its output pipe.
// Lifetime: Each child exits after its report or a requested failure.
#ifndef AOTX_CTRL_TESTS_INSPECT_FIXTURE_HPP
#define AOTX_CTRL_TESTS_INSPECT_FIXTURE_HPP

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

namespace aotx::ctrl::test {

inline std::string escaped_source(const std::string &source)
{
    const char hex[] = "0123456789abcdef";
    std::string out;
    for (std::size_t i = 0; i < std::min(source.size(), std::size_t{512}); ++i) {
        const unsigned char c = static_cast<unsigned char>(source[i]);
        if (c > 32 && c < 127 && c != '\\') out += static_cast<char>(c);
        else { out += "\\x"; out += hex[c / 16]; out += hex[c % 16]; }
    }
    if (source.size() > 512) out += "[cut]";
    return out;
}

inline std::string report(const std::string &source, unsigned id, bool supported)
{
    std::ostringstream out;
    out << "file=" << escaped_source(source) << '\n'
        << "architecture=family_" << id << '\n'
        << "rotary_pairs=split supported=yes\n"
        << "pre_tokenizer=tokenizer_" << id << " supported=yes\n"
        << "tokenizer_model=gpt2 supported=yes\n"
        << "tensors=" << id + 100 << '\n'
        << "block_type=Q8_0 id=" << id + 10 << " count=" << id + 100 << " supported=yes\n"
        << "layers=" << id + 1 << " hidden=" << id + 128 << " vocabulary=" << id + 1000 << '\n'
        << "layer_type=attention count=" << id + 1 << '\n'
        << "layer_sets_supported=" << (supported ? "yes" : "no")
        << " unknown_tensors=0 layer_limit=256\n"
        << "chat_template_bytes=" << id + 200 << " chat_template_sha256="
        << std::string(64, "0123456789abcdef"[id % 16]) << '\n'
        << "file_bytes=" << id + 100000 << "\nheader_bytes=" << id + 10000 << '\n';
    if (source.compare(0, 8, "https://") == 0) out << "received_bytes=" << id + 20000 << '\n';
    out << "build_support=" << (supported ? "yes" : "no") << "\nrun_verified=no\n"
        << "The support result covers the listed header fields and tensor sets only.\n"
        << "The header does not prove weight integrity, memory fit, wrap, prefill, or restore.\n";
    if (!supported)
        out << "This build cannot run this file with the listed unsupported or missing fields.\n";
    return out.str();
}

inline bool write_all(int fd, const std::string &text)
{
    std::size_t at = 0;
    while (at < text.size()) {
        const ssize_t wrote = ::write(fd, text.data() + at, text.size() - at);
        if (wrote > 0) at += static_cast<std::size_t>(wrote);
        else if (wrote < 0 && errno == EINTR) continue;
        else return false;
    }
    return true;
}

inline void replace(std::string &text, const std::string &old, const std::string &value)
{
    const auto at = text.find(old);
    if (at == std::string::npos) std::abort();
    text.replace(at, old.size(), value);
}

inline void replace_line(std::string &text, const std::string &prefix, const std::string &value)
{
    const auto at = text.find(prefix);
    if (at == std::string::npos) std::abort();
    text.replace(at, text.find('\n', at) - at + 1, value);
}

inline int inspect_child(const std::string &source)
{
    std::string mode;
    unsigned id = 0;
    if (source.compare(0, 8, "https://") == 0) {
        const auto marker = source.find("/case-");
        if (marker == std::string::npos) return 4;
        id = static_cast<unsigned>(std::stoul(source.substr(marker + 6)));
        mode = "good";
    } else {
        std::ifstream file(source);
        if (!(file >> mode >> id)) return 4;
    }
    std::string text = report(source, id, mode.compare(0, 11, "unsupported") != 0);
    if (mode == "missing_layer" || mode == "unsupported_layers") replace_line(text, "layer_type=", "");
    if (mode == "short_layers")
        replace_line(text, "layer_type=", "layer_type=attention count=" + std::to_string(id) + "\n");
    if (mode == "hang") for (;;) ::pause();
    if (mode == "empty") return 0;
    if (mode == "partial") text.pop_back();
    if (mode == "missing_support") replace_line(text, "build_support=", "");
    if (mode == "duplicate_support") replace(text, "build_support=yes\n", "build_support=yes\nbuild_support=yes\n");
    if (mode == "run_yes") replace(text, "run_verified=no", "run_verified=yes");
    if (mode == "missing_run") replace_line(text, "run_verified=", "");
    if (mode == "wrong_file") replace_line(text, "file=", "file=another-file\n");
    if (mode == "bad_support") replace(text, "build_support=yes", "build_support=maybe");
    if (mode == "overflow") replace_line(text, "tensors=", "tensors=18446744073709551616\n");
    if (mode == "negative") replace_line(text, "tensors=", "tensors=-1\n");
    if (mode == "number_tail") replace_line(text, "tensors=", "tensors=100x\n");
    if (mode == "bad_digest") replace(text, "chat_template_sha256=", "chat_template_sha256=g");
    if (mode == "duplicate_block") {
        const auto at = text.find("block_type=");
        text.insert(at, text.substr(at, text.find('\n', at) - at + 1));
    }
    if (mode == "duplicate_layer") {
        const auto at = text.find("layer_type=");
        text.insert(at, text.substr(at, text.find('\n', at) - at + 1));
    }
    if (mode == "missing_block") replace_line(text, "block_type=", "");
    if (mode == "block_overflow") replace_line(text, "block_type=", "block_type=Q8_0 id=4294967296 count=100 supported=yes\n");
    if (mode == "layer_overflow") replace_line(text, "layers=", "layers=4294967296 hidden=128 vocabulary=1000\n");
    if (mode == "bad_escape") replace_line(text, "architecture=", "architecture=bad\\tail\n");
    if (mode == "control") replace_line(text, "architecture=", "architecture=bad\033[2J\n");
    if (mode == "extra_field") text += "build_support=yes\n";
    if (mode == "received_local") replace(text, "build_support=", "received_bytes=100\nbuild_support=");
    if (mode == "late_tail") {
        const auto cut = text.size() - 35;
        if (!write_all(1, text.substr(0, cut))) return 5;
        ::usleep(10000);
        if (!write_all(1, text.substr(cut))) return 5;
        return 0;
    }
    if (mode == "oversize") text += std::string(131073, 'x');
    if (mode == "held_pipe") {
        const pid_t descendant = ::fork();
        if (descendant < 0) return 6;
        if (descendant == 0) for (;;) ::pause();
    }
    if (!write_all(1, text)) return 5;
    if (mode == "stderr") { if (!write_all(2, "The header read failed.\n")) return 5; }
    if (mode == "signal") { ::raise(SIGTERM); return 6; }
    if (mode == "exit2") return 2;
    if (mode == "exit1" || mode == "unsupported1") return 1;
    return 0;
}

} // namespace aotx::ctrl::test
#endif
