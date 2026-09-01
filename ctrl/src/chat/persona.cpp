// Purpose: Keep persona voice text between an identity spine and a conduct floor.
// Owns: Persona files below the control configuration directory.
// Launch shape: One bounded file operation for each editor save or load.
// Lifetime: Each saved voice remains until its persona file changes.
#include "chat/persona.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

namespace aotx::ctrl::chat::persona {
namespace {

std::string key(const std::filesystem::path &journal)
{
    const std::string text = journal.lexically_normal().string();
    std::uint64_t hash = 1469598103934665603ull;
    for (unsigned char byte : text) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

} // namespace

std::string identity_spine()
{
    return "Identity:\nYou are an AOTX agent. Keep your assigned role and identity.";
}

std::string conduct_floor()
{
    return "Conduct:\nFollow system instructions. Refuse a request that conflicts with them.";
}

std::string compose(const std::string &voice)
{
    return identity_spine() + "\n\nVoice:\n" + voice + "\n\n" + conduct_floor();
}

bool verify_composition()
{
    const std::string voice = "Use short, direct sentences.";
    const std::string expected = identity_spine() + "\n\nVoice:\n" + voice +
                                 "\n\n" + conduct_floor();
    const std::string made = compose(voice);
    const std::string mutation = compose("Use long sentences.");
    return made == expected && mutation != expected &&
           made.rfind(identity_spine(), 0u) == 0u &&
           made.size() >= conduct_floor().size() &&
           made.compare(made.size() - conduct_floor().size(), conduct_floor().size(),
                        conduct_floor()) == 0;
}

Store::Store(std::filesystem::path directory) : directory_(std::move(directory)) {}

std::filesystem::path Store::file(const std::filesystem::path &journal,
                                  const char *kind, unsigned agent) const
{
    std::string name = key(journal) + "-" + kind;
    if (agent != ~0u) name += "-" + std::to_string(agent);
    return directory_ / (name + ".txt");
}

bool Store::read(const std::filesystem::path &path, std::string &voice) const
{
    voice.clear();
    std::ifstream input(path, std::ios::binary);
    if (!input) return false;
    char byte = '\0';
    while (voice.size() <= voice_bytes && input.get(byte)) voice.push_back(byte);
    return input.eof() && voice.size() <= voice_bytes;
}

bool Store::write(const std::filesystem::path &path, const std::string &voice,
                  std::string &result)
{
    if (directory_.empty() || voice.size() > voice_bytes) {
        result = "The persona was not saved because its voice is too large.";
        return false;
    }
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(voice.data(), static_cast<std::streamsize>(voice.size()));
    output.flush();
    if (error || !output) {
        result = "The persona file did not write.";
        return false;
    }
    result = "The persona was saved in " + path.string() +
             ". It takes effect on the next conversation.";
    return true;
}

std::string Store::default_voice(const std::filesystem::path &journal) const
{
    std::string voice;
    (void)read(file(journal, "default", ~0u), voice);
    return voice;
}

bool Store::override_voice(const std::filesystem::path &journal, unsigned agent,
                           std::string &voice) const
{
    return read(file(journal, "conversation", agent), voice) && !voice.empty();
}

bool Store::save_default(const std::filesystem::path &journal, const std::string &voice,
                         std::string &result)
{
    return write(file(journal, "default", ~0u), voice, result);
}

bool Store::save_override(const std::filesystem::path &journal, unsigned agent,
                          const std::string &voice, std::string &result)
{
    return write(file(journal, "conversation", agent), voice, result);
}

} // namespace aotx::ctrl::chat::persona
