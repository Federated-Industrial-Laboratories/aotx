// Purpose: Draw simulated start, authorization, and reply controls.
// Owns: Control actions and their result notifications.
// Launch shape: One window updates one selected instance per action.
// Lifetime: Each action changes the simulated state immediately.
#include "control/control.hpp"

#include "imgui.h"
#include "theme/theme.hpp"

#include <charconv>

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
    sim::Conversation &conversation = state.conversations[state.selected_conversation];
    ImGui::Text("Conversation: %s", conversation.name.c_str());
    int reply = static_cast<int>(conversation.reply_bound);
    if (ImGui::SliderInt("Reply bound", &reply, 1, 8191)) {
        conversation.reply_bound = static_cast<unsigned>(reply);
    }
    ImGui::Checkbox("Auto-continue", &conversation.auto_continue);
    int pages = static_cast<int>(state.page_limit);
    if (ImGui::SliderInt("Agent pages", &pages, 16, 160)) {
        state.page_limit = static_cast<unsigned>(pages);
    }
    ImGui::TextDisabled("The profile permits 160 pages.");
    ImGui::End();
}

void draw(LiveState &view, instances::Lifecycle &lifecycle, replica::State &state,
          client::Client &client,
          toast::Lane &toasts, double now, bool *open)
{
    if (!view.initialized) {
        std::string value;
        if (replica::setting_value(state.settings(), "decode.reply_limit", value)) {
            std::from_chars(value.data(), value.data() + value.size(), view.reply_bound);
        }
        if (replica::setting_value(state.settings(), "decode.auto_continue", value)) {
            view.auto_continue = value == "1";
        }
        view.initialized = true;
    }
    if (!ImGui::Begin("Control", open)) {
        ImGui::End();
        return;
    }
    const std::vector<instances::LiveInstance> items = lifecycle.instances();
    const std::size_t selected = lifecycle.selected();
    const char *name = selected < items.size() ? items[selected].definition.name.c_str()
                                               : "No instance";
    ImGui::Text("Selected: %s", name);
    ImGui::Text("State: %s", state.phase().c_str());
    if (ImGui::Button("Start")) {
        if (!lifecycle.start(selected)) {
            toasts.add(lifecycle.refusal(), toast::Severity::error, now);
        }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Start a new run from the settings file.");
    ImGui::SameLine();
    if (ImGui::Button("Restore")) {
        if (!lifecycle.start(selected, true)) {
            toasts.add(lifecycle.refusal(), toast::Severity::error, now);
        }
    }
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Replay the latest saved journal before new input.");
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
        if (!lifecycle.stop(selected)) {
            toasts.add(lifecycle.refusal(), toast::Severity::error, now);
        }
    }
    ImGui::SeparatorText("Authorization queue");
    bool shown = false;
    for (const replica::PendingRequest &item : state.pending_requests()) {
        shown = true;
        ImGui::PushID(static_cast<int>(item.request));
        ImGui::Text("%llu  agent %llu calls %s",
                    static_cast<unsigned long long>(item.request),
                    static_cast<unsigned long long>(item.agent), item.tool.c_str());
        ImGui::TextDisabled("%s", item.path.c_str());
        if (ImGui::Button("Grant")) {
            client.send_line("authorize " + std::to_string(item.request));
        }
        ImGui::SameLine();
        if (ImGui::Button("Refuse")) {
            client.send_line("refuse " + std::to_string(item.request));
        }
        ImGui::Separator();
        ImGui::PopID();
    }
    if (!shown) ImGui::TextDisabled("No authorization requests are pending.");

    ImGui::SeparatorText("Reply controls");
    ImGui::SliderInt("Reply bound", &view.reply_bound, 1, 8191);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        client.send_line("set decode.reply_limit " + std::to_string(view.reply_bound));
    }
    if (ImGui::Checkbox("Auto-continue", &view.auto_continue)) {
        client.send_line(std::string("set decode.auto_continue ") +
                         (view.auto_continue ? "1" : "0"));
    }
    ImGui::InputInt("Agent pages", &view.pages);
    if (ImGui::Button("Set")) {
        if (state.agents().empty()) {
            toasts.add("The page change was refused because no agent is active.",
                       toast::Severity::error, now);
        } else if (view.pages < 0 || view.pages > 4096) {
            toasts.add("The page change was refused because the value is outside its range.",
                       toast::Severity::error, now);
        } else {
            const std::string value = view.pages == 0 ? "auto" : std::to_string(view.pages);
            const std::size_t selected_agent = state.selected_agent();
            client.send_line("agent " + std::to_string(state.agents()[selected_agent].id) +
                             " pages " + value);
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Set the agent page limit. Zero uses the profile limit.");
    ImGui::End();
}

} // namespace aotx::ctrl::control
