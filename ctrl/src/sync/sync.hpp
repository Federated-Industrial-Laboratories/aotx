// Purpose: Define disk-to-device synchronization rows and state.
// Owns: Last import times observed by the control program.
// Launch shape: One interface thread updates and draws one Sync panel.
// Lifetime: Row state exists while its instance binding remains selected.
#ifndef AOTX_CTRL_SYNC_HPP
#define AOTX_CTRL_SYNC_HPP

#include "client/client.hpp"
#include "replica/replica.hpp"

#include <filesystem>
#include <map>
#include <string>

namespace aotx::ctrl::sync {

enum class DiskState { in_step, disk_newer };

DiskState disk_state(std::filesystem::file_time_type disk,
                     std::filesystem::file_time_type last_import);
const char *state_word(bool known, DiskState state);
bool verify_state_logic();

struct State {
    struct Stamp {
        std::filesystem::file_time_type last_import{};
        bool known = false;
    };
    std::map<std::string, Stamp> stamps;
    std::size_t notes_seen = 0u;
    bool notes_bound = false;
    std::string result;
};

void draw(State &view, replica::State &state, client::Client &client, bool *open);

} // namespace aotx::ctrl::sync

#endif
