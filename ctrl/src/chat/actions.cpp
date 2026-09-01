// Purpose: Stop and export one live conversation.
// Owns: A new export file for each successful export act.
// Launch shape: One host act runs for one selected conversation.
// Lifetime: An export file remains on disk after the program ends.
#include "chat/actions.hpp"

#include <cctype>
#include <fstream>

namespace aotx::ctrl::chat {
namespace {

std::string safe_name(const std::string &name)
{
    std::string out;
    for (unsigned char byte : name) {
        if (std::isalnum(byte)) out.push_back(static_cast<char>(std::tolower(byte)));
        else if (!out.empty() && out.back() != '-') out.push_back('-');
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    return out.empty() ? "conversation" : out;
}

const char *speaker(const replica::TranscriptEvent &event)
{
    if (event.kind == "line") return "You";
    if (event.kind == "call") return "Tool call";
    if (event.kind == "grant" || event.kind == "refuse") return "Tool result";
    return "Agent";
}

} // namespace

std::string stop_command(unsigned agent)
{
    return "agent " + std::to_string(agent) + " stop";
}

bool export_conversation(const std::filesystem::path &directory,
                         const replica::Agent &agent, std::filesystem::path &written,
                         std::string &result)
{
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        result = "The conversation export directory did not create.";
        return false;
    }
    const std::string stem = safe_name(agent.conversation) + "-agent-" +
                             std::to_string(agent.id);
    unsigned suffix = 0u;
    do {
        written = directory / (stem + (suffix == 0u ? "" : "-" + std::to_string(suffix)) +
                               ".txt");
        ++suffix;
    } while (std::filesystem::exists(written, error) && !error);
    std::ofstream output(written);
    output << agent.conversation << "\n\n";
    for (const replica::TranscriptEvent &event : agent.transcript) {
        if (event.kind == "selection") continue;
        output << speaker(event) << ": ";
        if (!event.text.empty()) output << event.text;
        else if (event.kind == "done" && event.status == "stopped") output << "Reply stopped.";
        else output << event.kind << " " << event.status;
        output << "\n\n";
    }
    output.flush();
    if (!output) {
        result = "The conversation export file did not write.";
        return false;
    }
    result = "The conversation was exported to " + written.string() + ".";
    return true;
}

} // namespace aotx::ctrl::chat
