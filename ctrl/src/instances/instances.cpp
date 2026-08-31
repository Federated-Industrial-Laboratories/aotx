// Purpose: Present simulated instance cards and state actions.
// Owns: Tab actions and their result notifications.
// Launch shape: One tab is active in one instance window.
// Lifetime: Actions update the simulated state immediately.
#include "instances/instances.hpp"

#include "imgui.h"
#include "theme/theme.hpp"

#include <cstring>
#include <utility>

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

View::View() { std::strcpy(new_name.data(), "New instance"); }

void draw(View &view, sim::State &state, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Instances", open)) {
        ImGui::End();
        return;
    }
    if (state.instance_creation_requested) {
        view.create_visible = true;
        state.instance_creation_requested = false;
    }
    if (ImGui::Button("New instance")) view.create_visible = true;
    if (view.create_visible) {
        ImGui::InputText("Name", view.new_name.data(), view.new_name.size());
        if (ImGui::Button("Create instance")) {
            const std::string name = view.new_name.data();
            if (state.create_instance(name)) {
                toasts.add(name + " was created.", toast::Severity::success, now);
                std::strcpy(view.new_name.data(), "New instance");
                view.create_visible = false;
            } else {
                toasts.add(state.refusal(), toast::Severity::error, now);
            }
        }
        ImGui::Separator();
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

void draw(View &view, Lifecycle &lifecycle, toast::Lane &toasts, double now, bool *open)
{
    if (!ImGui::Begin("Instances", open)) {
        ImGui::End();
        return;
    }
    if (ImGui::Button("New instance")) view.create_visible = true;
    if (view.create_visible) {
        ImGui::InputText("Name", view.new_name.data(), view.new_name.size());
        ImGui::InputText("Journal directory", view.journal.data(), view.journal.size());
        ImGui::InputText("Settings file", view.settings.data(), view.settings.size());
        ImGui::InputText("Build directory", view.build.data(), view.build.size());
        ImGui::InputText("Model directory", view.models.data(), view.models.size());
        ImGui::InputInt("Card", &view.card);
        ImGui::TextDisabled("Enter the zero-based card number.");
        if (ImGui::Button("Create instance")) {
            Definition definition;
            definition.name = view.new_name.data();
            definition.journal = view.journal.data();
            definition.settings = view.settings.data();
            definition.build = view.build.data();
            definition.models = view.models.data();
            definition.card = view.card < 0 ? 0u : static_cast<unsigned>(view.card);
            if (view.card < 0 || !lifecycle.create(std::move(definition))) {
                const std::string refusal = view.card < 0
                    ? "The instance creation was refused because the card is not valid."
                    : lifecycle.refusal();
                toasts.add(refusal, toast::Severity::error, now);
            } else {
                view.create_visible = false;
                std::strcpy(view.new_name.data(), "New instance");
            }
        }
        ImGui::Separator();
    }
    const std::vector<LiveInstance> items = lifecycle.instances();
    if (ImGui::BeginTabBar("Live instance tabs")) {
        for (std::size_t index = 0u; index < items.size(); ++index) {
            const LiveInstance &item = items[index];
            if (!ImGui::BeginTabItem(item.definition.name.c_str())) continue;
            const ImVec4 color = item.state == LiveState::running ? theme::palette().running :
                                 item.state == LiveState::attaching ? theme::palette().attaching :
                                                                     theme::palette().stopped;
            ImGui::TextColored(color, "%s", state_name(item.state));
            ImGui::TextColored(color, "Phase: %s", item.phase.c_str());
            ImGui::TextColored(color, "Socket: %s", item.connection.c_str());
            ImGui::TextWrapped("%s", item.result.c_str());
            ImGui::TextDisabled("Card %u", item.definition.card);
            if (ImGui::Button("Start")) {
                if (!lifecycle.start(index)) {
                    toasts.add(lifecycle.refusal(), toast::Severity::error, now);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Stop")) {
                if (!lifecycle.stop(index)) {
                    toasts.add(lifecycle.refusal(), toast::Severity::error, now);
                }
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}

} // namespace aotx::ctrl::instances
