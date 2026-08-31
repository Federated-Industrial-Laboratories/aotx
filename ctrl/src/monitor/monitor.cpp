// Purpose: Draw simulated tick, ring, memory, and agent metrics.
// Owns: Palette cues for metric state.
// Launch shape: One panel draws all current monitor values.
// Lifetime: Values update from the simulated state each frame.
#include "monitor/monitor.hpp"

#include "imgui.h"
#include "theme/theme.hpp"

namespace aotx::ctrl::monitor {
namespace {

ImVec4 state_color(const std::string &state)
{
    if (state == "run" || state == "idle") return theme::palette().running;
    if (state == "prompt" || state == "tool") return theme::palette().attaching;
    return theme::palette().stopped;
}

} // namespace

void draw(const sim::State &state, bool *open)
{
    if (!ImGui::Begin("Monitor", open)) {
        ImGui::End();
        return;
    }
    const ImVec4 tick_color = state.tick_rate_hz >= 95.0f
                                  ? theme::palette().running
                                  : theme::palette().severity_warning;
    ImGui::TextColored(tick_color, "Tick rate %.1f Hz", state.tick_rate_hz);
    const ImVec4 ring_color = state.ring_occupancy < 0.75f
                                  ? theme::palette().running
                                  : theme::palette().severity_error;
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ring_color);
    ImGui::ProgressBar(state.ring_occupancy, ImVec2(-1.0f, 0.0f), "Ring occupancy");
    ImGui::PopStyleColor();
    ImGui::SeparatorText("Card memory");
    for (const sim::Instance &instance : state.instances) {
        for (const sim::Card &card : instance.cards) {
            ImGui::Text("%s / %s", instance.name.c_str(), card.name.c_str());
            const float use = card.memory_total_mib == 0
                                  ? 0.0f
                                  : static_cast<float>(card.memory_used_mib) /
                                        static_cast<float>(card.memory_total_mib);
            const ImVec4 color = use < 0.8f ? theme::palette().running
                                            : theme::palette().severity_error;
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, color);
            ImGui::ProgressBar(use, ImVec2(-1.0f, 0.0f));
            ImGui::PopStyleColor();
            ImGui::TextDisabled("%u MiB of %u MiB", card.memory_used_mib,
                                card.memory_total_mib);
        }
    }
    ImGui::SeparatorText("Agents");
    for (const sim::Agent &agent : state.agents) {
        ImGui::TextColored(state_color(agent.state), "%s  %s  %s  %u pages",
                           agent.name.c_str(), agent.role.c_str(), agent.state.c_str(), agent.pages);
    }
    ImGui::End();
}

} // namespace aotx::ctrl::monitor
