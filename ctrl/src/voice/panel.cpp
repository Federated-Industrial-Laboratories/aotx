// Purpose: Draw voice switches, dials, assignments, and a sample control.
// Owns: Immediate settings edits for the active voice queue.
// Launch shape: One interface frame applies each changed control.
// Lifetime: No panel-local copy outlives the current frame.
#include "voice/panel.hpp"

#include "imgui.h"
#include "voice/voice.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace aotx::ctrl::voice {
namespace {

void category_switch(Queue &queue, const char *label, Category category, bool value)
{
    bool changed = value;
    if (ImGui::Checkbox(label, &changed)) queue.set_category(category, changed);
}

} // namespace

void draw(Queue &queue, bool *open)
{
    if (!ImGui::Begin("Voice", open)) {
        ImGui::End();
        return;
    }
    Controls values = queue.controls();
    bool master = values.master;
    if (ImGui::Checkbox("Master", &master)) queue.set_master(master);
    category_switch(queue, "Replies", Category::reply, values.replies);
    category_switch(queue, "Toasts", Category::toast, values.toasts);
    category_switch(queue, "Tool calls", Category::tool, values.tools);
    category_switch(queue, "Lifecycle", Category::lifecycle, values.lifecycle);
#ifdef AOTX_AFFECT
    bool coupling = values.coupling;
    if (ImGui::Checkbox("Couple", &coupling)) queue.set_coupling(coupling);
    ImGui::SameLine();
    ImGui::TextDisabled("The spoken voice coupling uses arousal gains -0.25 for length and +0.167 for noise, with caps 0.75 to 1.25 and 0.50 to 0.834.");
    ImGui::TextDisabled("It uses valence gain +0.10 for noise width, with caps 0.70 to 0.90.");
#endif

    float rate = values.rate;
    if (ImGui::SliderFloat("Speech rate", &rate, 0.5f, 2.0f, "%.2fx")) {
        queue.set_rate(rate);
    }
    int depth = static_cast<int>(values.depth);
    if (ImGui::SliderInt("Queue depth", &depth, 1, 32)) {
        queue.set_depth(static_cast<std::size_t>(depth));
    }
    if (ImGui::Button("Test")) queue.test();
    ImGui::SameLine();
    ImGui::TextDisabled("Speak one sample line with the current settings.");

    ImGui::SeparatorText("Agent voices");
    const std::vector<std::filesystem::path> voices = queue.voices();
    if (voices.empty()) {
        ImGui::TextWrapped("%s", queue.refusal().c_str());
        ImGui::End();
        return;
    }
    if (ImGui::BeginTable("Agent voice assignments", 2,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Agent");
        ImGui::TableSetupColumn("Voice");
        ImGui::TableHeadersRow();
        for (std::size_t agent = 0u; agent < queue.agent_count(); ++agent) {
            const std::size_t assigned = std::min(queue.agent_assignment(agent),
                                                  voices.size() - 1u);
            const std::string agent_name = "Agent " + std::to_string(agent);
            ImGui::PushID(static_cast<int>(agent));
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextUnformatted(agent_name.c_str());
            ImGui::TableSetColumnIndex(1);
            if (ImGui::BeginCombo("##voice", voices[assigned].stem().c_str())) {
                for (std::size_t voice = 0u; voice < voices.size(); ++voice) {
                    if (ImGui::Selectable(voices[voice].stem().c_str(), voice == assigned)) {
                        queue.set_agent_assignment(agent, voice);
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

} // namespace aotx::ctrl::voice
