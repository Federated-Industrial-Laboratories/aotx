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

} // namespace aotx::ctrl::browser
