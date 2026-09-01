// Purpose: Read the model catalog, local store, and active manifest.
// Owns: Nothing; the caller owns model view rows.
// Launch shape: One interface thread reads one set of store files.
// Lifetime: Rows remain in caller-owned state until the next read.
#ifndef AOTX_CTRL_REPLICA_STORE_HPP
#define AOTX_CTRL_REPLICA_STORE_HPP

#include "replica/replica.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace aotx::ctrl::replica::store {

bool read(const std::filesystem::path &catalog, const std::filesystem::path &directory,
          std::vector<Model> &models, std::string &reason);
bool read_controls(const std::filesystem::path &directory,
                   std::vector<ModelParameters> &parameters,
                   std::vector<SteerVector> &vectors,
                   std::vector<VoiceProfile> &profiles, std::string &reason);

} // namespace aotx::ctrl::replica::store

#endif
