// Purpose: Define typed mirror and run-time card telemetry readers.
// Owns: Mirror mappings and the optional NVML library handle.
// Launch shape: One interface thread samples one mirror and all cards.
// Lifetime: Readers release mappings and the library at program exit.
#ifndef AOTX_CTRL_TELEMETRY_HPP
#define AOTX_CTRL_TELEMETRY_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aotx::ctrl::monitor {

struct MirrorSample {
    std::uint64_t sequence = 0u;
    std::uint64_t tick = 0u;
    double tick_rate = 0.0;
    bool available = false;
    std::string result = "The mirror is not attached.";
};

struct CardMemory {
    unsigned index = 0u;
    std::string name;
    std::uint64_t used_mib = 0u;
    std::uint64_t total_mib = 0u;
};

class Telemetry {
  public:
    Telemetry();
    ~Telemetry();
    Telemetry(const Telemetry &) = delete;
    Telemetry &operator=(const Telemetry &) = delete;

    void tick(int mirror_descriptor, double now);
    const MirrorSample &mirror() const;
    const std::vector<CardMemory> &cards() const;
    const std::string &card_result() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aotx::ctrl::monitor

#endif
