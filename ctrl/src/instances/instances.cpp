// Purpose: Present simulated instance cards and state actions.
// Owns: Tab actions and their result notifications.
// Launch shape: One tab is active in one instance window.
// Lifetime: Actions update the simulated state immediately.
#include "instances/instances.hpp"

#include "imgui.h"
#include "theme/theme.hpp"

namespace aotx::ctrl::instances {
namespace {

ImVec4 state_color(sim::InstanceState state)
{
    const theme::Palette &colors = theme::palette();
    switch (state) {
    case sim::InstanceState::running: return colors.running;
    case sim::InstanceState::attaching: return colors.attaching;
    case sim::InstanceState::stopped: return colors.stopped;
    }
    return colors.stopped;
}

void add_result(toast::Lane &toasts, const sim::Instance &instance,
                sim::InstanceState state, double now)
{
    if (state == sim::InstanceState::running) {
        toasts.add(instance.name + " started.", toast::Severity::success, now);
    } else if (state == sim::InstanceState::attaching) {
        toasts.add(instance.name + " is attaching.", toast::Severity::info, now);
    } else {
        toasts.add(instance.name + " stopped.", toast::Severity::warning, now);
    }
}

void draw_card(const sim::Instance &instance)
{
    ImGui::TextColored(state_color(instance.state), "%s", sim::state_name(instance.state));
    for (const sim::Card &card : instance.cards) {
        ImGui::Text("%s", card.name.c_str());
        ImGui::TextDisabled("%u MiB of %u MiB", card.memory_used_mib, card.memory_total_mib);
        const float progress = card.memory_total_mib == 0
                                   ? 0.0f
                                   : static_cast<float>(card.memory_used_mib) /
                                         static_cast<float>(card.memory_total_mib);
        ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f));
    }
}

} // namespace

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Instances", open)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginTabBar("Instance tabs")) {
        for (std::size_t index = 0; index < state.instances.size(); ++index) {
            sim::Instance &instance = state.instances[index];
            const ImGuiTabItemFlags flags = state.instance_selection_requested &&
                                                    state.selected_instance == index
                                                ? ImGuiTabItemFlags_SetSelected
                                                : ImGuiTabItemFlags_None;
            if (ImGui::BeginTabItem(instance.name.c_str(), nullptr, flags)) {
                state.selected_instance = index;
                draw_card(instance);
                ImGui::Spacing();
                if (ImGui::Button("Start")) {
                    state.set_instance_state(index, sim::InstanceState::running, now);
                    add_result(toasts, instance, sim::InstanceState::running, now);
                }
                ImGui::SameLine();
                if (ImGui::Button("Attach")) {
                    state.set_instance_state(index, sim::InstanceState::attaching, now);
                    add_result(toasts, instance, sim::InstanceState::attaching, now);
                }
                ImGui::SameLine();
                if (ImGui::Button("Stop")) {
                    state.set_instance_state(index, sim::InstanceState::stopped, now);
                    add_result(toasts, instance, sim::InstanceState::stopped, now);
                }
                ImGui::EndTabItem();
            }
        }
        state.instance_selection_requested = false;
        ImGui::EndTabBar();
    }
    ImGui::End();
}

} // namespace aotx::ctrl::instances
