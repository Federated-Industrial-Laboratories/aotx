// Purpose: List stored runs and read one simulated transcript.
// Owns: Stored run selection controls.
// Launch shape: One panel draws one selected transcript.
// Lifetime: Stored transcript data remains in the simulated state.
#include "browser/browser.hpp"

#include "imgui.h"
#include "theme/theme.hpp"

namespace aotx::ctrl::browser {
namespace {

const char *role_name(sim::Role role)
{
    switch (role) {
    case sim::Role::user: return "You";
    case sim::Role::system: return "System";
    case sim::Role::agent: return "Agent";
    }
    return "System";
}

ImVec4 role_color(sim::Role role)
{
    if (role == sim::Role::user) return theme::palette().user_role;
    if (role == sim::Role::agent) return theme::palette().agent_role;
    return theme::palette().system_role;
}

} // namespace

void draw(State &view, const sim::State &state, bool *open)
{
    if (!ImGui::Begin("Transcripts", open)) {
        ImGui::End();
        return;
    }
    if (ImGui::BeginListBox("Past runs", ImVec2(180.0f, 120.0f))) {
        for (std::size_t index = 0; index < state.past_runs.size(); ++index) {
            if (ImGui::Selectable(state.past_runs[index].name.c_str(), view.selected == index)) {
                view.selected = index;
            }
        }
        ImGui::EndListBox();
    }
    if (view.selected < state.past_runs.size()) {
        const sim::PastRun &run = state.past_runs[view.selected];
        ImGui::TextWrapped("%s", run.result.c_str());
        ImGui::Separator();
        for (const sim::TranscriptEvent &event : run.transcript) {
            ImGui::TextColored(role_color(event.role), "%s", role_name(event.role));
            ImGui::TextWrapped("%s", event.stated.c_str());
        }
    }
    ImGui::End();
}

void draw(State &view, const replica::State &state, bool *open)
{
    if (!ImGui::Begin("Transcripts", open)) {
        ImGui::End();
        return;
    }
    const std::vector<replica::Boot> &boots = state.boots();
    if (view.selected >= boots.size()) view.selected = 0u;
    if (ImGui::BeginListBox("Past boots", ImVec2(260.0f, 140.0f))) {
        for (std::size_t index = 0u; index < boots.size(); ++index) {
            const std::string phase = index == 0u ? state.phase() : "closed";
            const std::string label = boots[index].name + "  " + phase;
            if (ImGui::Selectable(label.c_str(), view.selected == index)) {
                view.selected = index;
                view.loaded_boot.clear();
            }
        }
        ImGui::EndListBox();
    }
    if (boots.empty()) {
        ImGui::TextColored(theme::palette().severity_warning,
                           "The journal has no past boot directory.");
        ImGui::End();
        return;
    }
    const replica::Boot &selected = boots[view.selected];
    if (view.loaded_boot != selected.name) {
        view.loaded_boot = selected.name;
        if (!replica::read_boot_transcripts(selected.directory, view.agents, view.result)) {
            view.agents.clear();
        }
    }
    const bool readable = !view.agents.empty();
    ImGui::TextColored(readable ? theme::palette().running
                                : theme::palette().severity_warning,
                       "%s", view.result.c_str());
    ImGui::Separator();
    for (const replica::Agent &agent : view.agents) {
        ImGui::SeparatorText(agent.conversation.c_str());
        for (const replica::TranscriptEvent &event : agent.transcript) {
            const bool user = event.kind == "line";
            const bool system = event.kind == "bound" || event.kind == "grant" ||
                                event.kind == "refuse" || event.kind == "verdict";
            const ImVec4 color = user ? theme::palette().user_role :
                                 system ? theme::palette().system_role :
                                          theme::palette().agent_role;
            ImGui::TextColored(color, "%s", event.kind.c_str());
            if (!event.tool.empty()) ImGui::TextWrapped("%s", event.tool.c_str());
            if (!event.text.empty()) ImGui::TextWrapped("%s", event.text.c_str());
        }
    }
    ImGui::End();
}

} // namespace aotx::ctrl::browser
