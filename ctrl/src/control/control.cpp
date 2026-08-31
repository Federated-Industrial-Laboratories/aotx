// Purpose: Draw simulated start, authorization, and reply controls.
// Owns: Control actions and their result notifications.
// Launch shape: One window updates one selected instance per action.
// Lifetime: Each action changes the simulated state immediately.
#include "control/control.hpp"

#include "imgui.h"
#include "theme/theme.hpp"

namespace aotx::ctrl::control {
namespace {

const char *answer_name(sim::AuthorizationState answer)
{
    switch (answer) {
    case sim::AuthorizationState::pending: return "pending";
    case sim::AuthorizationState::granted: return "granted";
    case sim::AuthorizationState::refused: return "refused";
    }
    return "refused";
}

void answer(sim::State &state, toast::Lane &toasts, std::size_t index,
            sim::AuthorizationState value, double now)
{
    const unsigned id = state.authorizations[index].id;
    if (state.answer_authorization(index, value)) {
        toasts.add("Authorization " + std::to_string(id) + " was " + answer_name(value) + ".",
                   value == sim::AuthorizationState::granted ? toast::Severity::success
                                                             : toast::Severity::warning,
                   now);
    } else {
        toasts.add("The authorization answer was refused.", toast::Severity::error, now);
    }
}

} // namespace

void draw(sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Control", open)) {
        ImGui::End();
        return;
    }
    sim::Instance &instance = state.instances[state.selected_instance];
    ImGui::Text("Selected: %s", instance.name.c_str());
    if (ImGui::Button("Start")) {
        state.set_instance_state(state.selected_instance, sim::InstanceState::running, now);
        toasts.add(instance.name + " started.", toast::Severity::success, now);
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        state.set_instance_state(state.selected_instance, sim::InstanceState::stopped, now);
        toasts.add(instance.name + " stopped.", toast::Severity::warning, now);
    }
    ImGui::SeparatorText("Authorization queue");
    for (std::size_t index = 0; index < state.authorizations.size(); ++index) {
        const sim::Authorization &item = state.authorizations[index];
        ImGui::PushID(static_cast<int>(item.id));
        ImGui::Text("%u  %s calls %s", item.id, item.agent.c_str(), item.tool.c_str());
        ImGui::TextDisabled("%s", item.argument.c_str());
        if (item.state == sim::AuthorizationState::pending) {
            if (ImGui::Button("Grant")) {
                answer(state, toasts, index, sim::AuthorizationState::granted, now);
            }
            ImGui::SameLine();
            if (ImGui::Button("Refuse")) {
                answer(state, toasts, index, sim::AuthorizationState::refused, now);
            }
        } else {
            ImGui::TextColored(item.state == sim::AuthorizationState::granted
                                   ? theme::palette().severity_success
                                   : theme::palette().severity_error,
                               "%s", answer_name(item.state));
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    ImGui::SeparatorText("Reply controls");
    int reply = static_cast<int>(state.reply_bound);
    if (ImGui::SliderInt("Reply bound", &reply, 1, 8191)) {
        state.reply_bound = static_cast<unsigned>(reply);
    }
    ImGui::Checkbox("Auto-continue", &state.auto_continue);
    int pages = static_cast<int>(state.page_limit);
    if (ImGui::SliderInt("Agent pages", &pages, 16, 160)) {
        state.page_limit = static_cast<unsigned>(pages);
    }
    ImGui::TextDisabled("The profile permits 160 pages.");
    ImGui::End();
}

} // namespace aotx::ctrl::control
