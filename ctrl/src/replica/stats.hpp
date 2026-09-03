// Purpose: Define incremental readers for statistics and measurement files.
// Owns: File cursors, statistics rows, and optional measurement rings.
// Launch shape: One interface thread tails each derived file.
// Lifetime: A reader follows the selected boot until it changes.
#ifndef AOTX_CTRL_REPLICA_STATS_HPP
#define AOTX_CTRL_REPLICA_STATS_HPP

#include "replica/replica.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace aotx::ctrl::replica::stats {

class Reader {
  public:
    Reader();
    ~Reader();
    Reader(const Reader &) = delete;
    Reader &operator=(const Reader &) = delete;

    void read(const std::filesystem::path &boot, const std::filesystem::path &store, double now,
              std::vector<std::string> &results);
    const std::vector<TokenStat> &tokens() const;
    const std::vector<PageStat> &pages() const;
#ifdef AOTX_AFFECT
    const std::vector<AffectTrace> &affect_traces() const;
    const std::vector<QualityLine> &quality_lines() const;
    const std::optional<Calibration> &calibration() const;
    const std::vector<ProbeAccuracy> &probe_accuracies() const;
#endif
    double token_rate() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aotx::ctrl::replica::stats

#endif
