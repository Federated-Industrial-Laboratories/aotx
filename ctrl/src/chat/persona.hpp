// Purpose: Compose and store contained persona voice text.
// Owns: Paths to one file for each saved persona.
// Launch shape: One interface thread reads or writes one persona at a time.
// Lifetime: The store path exists for the program lifetime.
#ifndef AOTX_CTRL_CHAT_PERSONA_HPP
#define AOTX_CTRL_CHAT_PERSONA_HPP

#include <filesystem>
#include <string>

namespace aotx::ctrl::chat::persona {

constexpr std::size_t voice_bytes = 8192u;
constexpr std::size_t overlay_bytes = 1536u;

struct RoleModule {
    std::string name;
    std::filesystem::path directory;
    std::string overlay;
};

std::string identity_spine();
std::string conduct_floor();
std::string compose(const std::string &voice);
bool verify_composition();
bool write_role_module(const std::filesystem::path &root, const std::string &voice,
                       RoleModule &module, std::string &result);

class Store {
  public:
    explicit Store(std::filesystem::path directory = {});
    std::string default_voice(const std::filesystem::path &journal) const;
    bool override_voice(const std::filesystem::path &journal, unsigned agent,
                        std::string &voice) const;
    bool save_default(const std::filesystem::path &journal, const std::string &voice,
                      std::string &result);
    bool save_override(const std::filesystem::path &journal, unsigned agent,
                       const std::string &voice, std::string &result);

  private:
    std::filesystem::path file(const std::filesystem::path &journal,
                               const char *kind, unsigned agent) const;
    bool read(const std::filesystem::path &path, std::string &voice) const;
    bool write(const std::filesystem::path &path, const std::string &voice,
               std::string &result);
    std::filesystem::path directory_;
};

} // namespace aotx::ctrl::chat::persona

#endif
