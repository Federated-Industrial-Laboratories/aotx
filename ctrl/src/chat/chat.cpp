// Purpose: Render simulated chat events and accept user text.
// Owns: Role cues, tool detail expansion, and editor key actions.
// Launch shape: One chat window processes one editor action per frame.
// Lifetime: Transcript data remains owned by the simulated state.
#include "chat/chat.hpp"

#include "imgui.h"
#include "theme/theme.hpp"

#include <cstring>

namespace aotx::ctrl::chat {
namespace {

ImVec4 role_color(sim::Role role)
{
    const theme::Palette &colors = theme::palette();
    switch (role) {
    case sim::Role::user: return colors.user_role;
    case sim::Role::system: return colors.system_role;
    case sim::Role::agent: return colors.agent_role;
    }
    return colors.system_role;
}

const char *role_name(sim::Role role)
{
    switch (role) {
    case sim::Role::user: return "You";
    case sim::Role::system: return "System";
    case sim::Role::agent: return "Agent";
    }
    return "System";
}

int editor_callback(ImGuiInputTextCallbackData *data)
{
    if (ImGui::GetIO().KeyAlt && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        data->InsertChars(data->CursorPos, "\n");
    }
    return 0;
}

bool draw_event(const sim::TranscriptEvent &event)
{
    bool continue_requested = false;
    ImGui::PushID(&event);
    ImGui::TextColored(role_color(event.role), "%s", role_name(event.role));
    if (event.kind == sim::EventKind::tool_call) {
        ImGui::TextUnformatted(event.stated.c_str());
        if (ImGui::TreeNode("Full call")) {
            ImGui::TextWrapped("%s", event.detail.c_str());
            ImGui::TreePop();
        }
    } else if (event.kind == sim::EventKind::reply_bound) {
        ImGui::TextColored(theme::palette().severity_warning, "%s", event.stated.c_str());
        if (ImGui::Button("Continue")) {
            continue_requested = true;
        }
    } else {
        ImGui::TextWrapped("%s%s", event.stated.c_str(), event.streaming ? "|" : "");
    }
    ImGui::Separator();
    ImGui::PopID();
    return continue_requested;
}

} // namespace

void draw(View &view, sim::State &state, double now, bool *open)
{
    if (!ImGui::Begin("Chat", open)) {
        ImGui::End();
        return;
    }

    const float editor_height = ImGui::GetTextLineHeightWithSpacing() * 5.0f;
    const float counter_height = ImGui::GetTextLineHeightWithSpacing() * 1.5f;
    ImGui::BeginChild("Transcript", ImVec2(0.0f, -(editor_height + counter_height)),
                      ImGuiChildFlags_Borders);
    bool continue_requested = false;
    for (const sim::TranscriptEvent &event : state.transcript) {
        continue_requested = draw_event(event) || continue_requested;
    }
    if (view.follow && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 8.0f) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    if (continue_requested) {
        state.continue_reply(now);
    }

    const ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                      ImGuiInputTextFlags_CallbackAlways;
    const bool entered = ImGui::InputTextMultiline("##editor", view.editor.data(),
                                                    view.editor.size(), ImVec2(-1.0f, editor_height),
                                                    flags, editor_callback);
    const bool alternate = ImGui::GetIO().KeyAlt;
    if (entered && !alternate) {
        state.send(view.editor.data(), now);
        view.editor.fill('\0');
        ImGui::SetKeyboardFocusHere(-1);
    }
    ImGui::TextDisabled("%zu bytes", std::strlen(view.editor.data()));
    ImGui::SameLine();
    ImGui::TextDisabled("Enter sends. Alt-Enter inserts a line.");
    ImGui::End();
}

} // namespace aotx::ctrl::chat
