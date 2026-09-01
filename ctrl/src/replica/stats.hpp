// Purpose: Define incremental readers for token and page statistics files.
// Owns: File cursors and the latest typed statistics rows.
// Launch shape: One interface thread tails both files.
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

    void read(const std::filesystem::path &boot, double now,
              std::vector<std::string> &results);
    const std::vector<TokenStat> &tokens() const;
    const std::vector<PageStat> &pages() const;
    double token_rate() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace aotx::ctrl::replica::stats

#endif
