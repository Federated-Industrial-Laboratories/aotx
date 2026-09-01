// Purpose: Read one bounded value from a control settings file.
// Owns: No state; the caller owns the returned text.
// Launch shape: One interface thread scans one settings file.
// Lifetime: Temporary line text ends when the call returns.
#include "replica/replica.hpp"

#include <fstream>

namespace aotx::ctrl::replica {
namespace {

constexpr std::size_t line_bound = 8192u;

std::string trim(std::string text)
{
    const auto space = [](unsigned char byte) {
        return byte == ' ' || byte == '\t' || byte == '\r';
    };
    while (!text.empty() && space(static_cast<unsigned char>(text.front()))) text.erase(0u, 1u);
    while (!text.empty() && space(static_cast<unsigned char>(text.back()))) text.pop_back();
    return text;
}

} // namespace

bool setting_value(const std::filesystem::path &path, const std::string &key,
                   std::string &value)
{
    std::ifstream file(path);
    std::string line;
    bool found = false;
    if (!file) return false;
    while (std::getline(file, line)) {
        if (line.size() > line_bound) return false;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const std::size_t equal = line.find('=');
        if (equal == std::string::npos || trim(line.substr(0u, equal)) != key) continue;
        value = trim(line.substr(equal + 1u));
        found = true;
    }
    return file.eof() && found;
}

} // namespace aotx::ctrl::replica
