// Purpose: Bind model header facts to the selected file and build.
// Owns: Inspection selection, file stamps, and details window state.
// Threading: One interface thread checks the binding before each display.
// Lifetime: Facts are cleared when their source binding changes.
#ifndef AOTX_CTRL_MODEL_DETAILS_HPP
#define AOTX_CTRL_MODEL_DETAILS_HPP

#include "model/inspect.hpp"
#include "replica/replica.hpp"

namespace aotx::ctrl::model {

struct DetailsState {
    InspectAction action;
    std::filesystem::path build_path, source;
    std::string selection, stamp, result;
    bool open = false;

    void start(const std::filesystem::path &build, const std::filesystem::path &directory,
               const replica::Model &row);
    void update(const std::filesystem::path &build, const std::filesystem::path &directory,
                const std::vector<replica::Model> &rows);
    void clear();
};

void draw_details(DetailsState &state);

} // namespace aotx::ctrl::model
#endif
