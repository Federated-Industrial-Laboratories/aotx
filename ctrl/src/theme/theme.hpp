// Purpose: Define the colors and style of the control program.
// Owns: The named palette and its ImGui style mapping.
// Launch shape: One user interface thread uses one palette.
// Lifetime: The palette is constant for the process lifetime.
#ifndef AOTX_CTRL_THEME_HPP
#define AOTX_CTRL_THEME_HPP

#include "imgui.h"

namespace aotx::ctrl::theme {

struct Palette {
    ImVec4 running;
    ImVec4 attaching;
    ImVec4 stopped;
    ImVec4 user_role;
    ImVec4 system_role;
    ImVec4 agent_role;
    ImVec4 severity_info;
    ImVec4 severity_success;
    ImVec4 severity_warning;
    ImVec4 severity_error;
};

const Palette &palette();
void apply();

} // namespace aotx::ctrl::theme

#endif
