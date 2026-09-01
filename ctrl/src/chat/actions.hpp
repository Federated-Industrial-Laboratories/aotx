// Purpose: Define conversation acts that do not depend on the graphics layer.
// Owns: No state; callers own paths and transcript rows.
// Launch shape: One interface or live-check thread runs one act.
// Lifetime: Returned text remains valid in caller-owned storage.
#ifndef AOTX_CTRL_CHAT_ACTIONS_HPP
#define AOTX_CTRL_CHAT_ACTIONS_HPP

#include "replica/replica.hpp"

#include <filesystem>
#include <string>

namespace aotx::ctrl::chat {

std::string stop_command(unsigned agent);
bool export_conversation(const std::filesystem::path &directory,
                         const replica::Agent &agent, std::filesystem::path &written,
                         std::string &result);

} // namespace aotx::ctrl::chat

#endif
