// Purpose: Render simulated or live chat events and accept user text.
// Owns: Role cues, tool detail expansion, and editor key actions.
// Launch shape: One chat window processes one editor action per frame.
// Lifetime: Transcript data remains owned by its selected data state.
#include "chat/chat.hpp"

#include "client/client.hpp"
#include "imgui.h"
#include "replica/replica.hpp"
#include "theme/theme.hpp"
#include "voice/voice.hpp"

#include <algorithm>
#include <cstring>
#include <string>

namespace aotx::ctrl::chat {
namespace {

const char *callback_insertion = nullptr;

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

ImGuiInputTextFlags editor_flags(bool alternate)
{
    ImGuiInputTextFlags flags = ImGuiInputTextFlags_EnterReturnsTrue |
                                ImGuiInputTextFlags_CallbackAlways;
    if (!alternate) flags |= ImGuiInputTextFlags_CtrlEnterForNewLine;
    return flags;
}

int editor_callback(ImGuiInputTextCallbackData *data)
{
    if (ImGui::GetIO().KeyAlt && ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
        data->InsertChars(data->CursorPos, "\n");
    }
    if (callback_insertion != nullptr) {
        data->InsertChars(data->CursorPos, callback_insertion);
        callback_insertion = nullptr;
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

bool live_user(const replica::TranscriptEvent &event)
{
    return event.kind == "line";
}

bool live_tool(const replica::TranscriptEvent &event)
{
    return event.kind == "call";
}

std::string stated_text(std::string text)
{
    for (char &byte : text) {
        if (static_cast<unsigned char>(byte) < 0x20u || byte == 0x7f) byte = ' ';
    }
    return text;
}

std::string stated_tool(unsigned agent, const replica::TranscriptEvent &event)
{
    std::string line = "Agent " + std::to_string(agent) + " calls " + event.tool;
    const std::string detail = stated_text(event.text);
    if (!detail.empty()) line += " " + detail;
    return line;
}

bool draw_live_event(const replica::TranscriptEvent &event, unsigned agent,
                     bool allow_continue)
{
    bool continue_requested = false;
    ImGui::PushID(&event);
    const sim::Role role = live_user(event) ? sim::Role::user : sim::Role::agent;
    ImGui::TextColored(role_color(role), "%s", role_name(role));
    if (live_tool(event)) {
        const std::string detail = stated_text(event.text);
        ImGui::TextUnformatted(stated_tool(agent, event).c_str());
        if (ImGui::TreeNode("Full call")) {
            ImGui::TextWrapped("%s", detail.c_str());
            ImGui::TreePop();
        }
    } else if (event.kind == "bound") {
        ImGui::TextColored(theme::palette().severity_warning,
                           "The reply is at its limit.");
        if (allow_continue && ImGui::Button("Continue")) continue_requested = true;
    } else if (!event.text.empty()) {
        ImGui::TextWrapped("%s%s", event.text.c_str(), event.kind == "part" ? "|" : "");
    } else {
        ImGui::TextWrapped("%s request %llu, %s", event.tool.c_str(),
                           static_cast<unsigned long long>(event.request), event.status.c_str());
    }
    ImGui::Separator();
    ImGui::PopID();
    return continue_requested;
}

void begin_editor_context()
{
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2(640.0f, 480.0f);
    io.DeltaTime = 1.0f / 60.0f;
    io.AddFocusEvent(true);
    unsigned char *pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
}

bool editor_frame(char *text, std::size_t size, ImGuiInputTextFlags flags, bool focus = false)
{
    ImGui::NewFrame();
    if (focus) ImGui::SetNextWindowFocus();
    ImGui::Begin("Editor check");
    if (focus) ImGui::SetKeyboardFocusHere();
    const bool entered = ImGui::InputTextMultiline(
        "##check", text, size, ImVec2(300.0f, 80.0f), flags, editor_callback);
    ImGui::End();
    ImGui::Render();
    return entered;
}

void focus_editor(char *text, std::size_t size)
{
    editor_frame(text, size, editor_flags(false), true);
    editor_frame(text, size, editor_flags(false));
    editor_frame(text, size, editor_flags(false));
}

bool run_key_path(bool alternate)
{
    begin_editor_context();
    std::array<char, 16> text{};
    std::memcpy(text.data(), "line", 5);
    focus_editor(text.data(), text.size());

    ImGuiIO &io = ImGui::GetIO();
    if (alternate) io.AddKeyEvent(ImGuiMod_Alt, true);
    io.AddKeyEvent(ImGuiKey_Enter, true);
    const bool entered = editor_frame(text.data(), text.size(), editor_flags(alternate));
    const bool result = alternate ? !entered && std::strcmp(text.data(), "\nline") == 0
                                  : entered && std::strcmp(text.data(), "line") == 0;
    ImGui::DestroyContext();
    return result;
}

bool run_under_limit_path()
{
    begin_editor_context();
    View view;
    std::memset(view.editor.data(), 'a', 3998);
    focus_editor(view.editor.data(), view.editor.size());
    ImGui::GetIO().AddInputCharacter('x');
    editor_frame(view.editor.data(), view.editor.size(), editor_flags(false));
    const bool result = std::strlen(view.editor.data()) == 3999;
    ImGui::DestroyContext();
    return result;
}

bool run_paste_limit_path()
{
    begin_editor_context();
    View view;
    std::memset(view.editor.data(), 'a', 3995);
    focus_editor(view.editor.data(), view.editor.size());
    ImGui::SetClipboardText("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
    ImGui::GetIO().AddKeyEvent(ImGuiMod_Ctrl, true);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_V, true);
    editor_frame(view.editor.data(), view.editor.size(), editor_flags(false));
    const bool result = std::strlen(view.editor.data()) == 4000;
    ImGui::DestroyContext();
    return result;
}

bool run_callback_limit_path()
{
    begin_editor_context();
    View view;
    std::memset(view.editor.data(), 'a', 3995);
    focus_editor(view.editor.data(), view.editor.size());
    callback_insertion = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    editor_frame(view.editor.data(), view.editor.size(), editor_flags(false));
    const bool result = std::strlen(view.editor.data()) == 4000;
    ImGui::DestroyContext();
    return result;
}

} // namespace

std::string window_name(const sim::Conversation &conversation, std::size_t index)
{
    return conversation.name + "##conversation-" + std::to_string(index);
}

std::string window_name(const replica::Agent &agent)
{
    return agent.conversation + "##agent-" + std::to_string(agent.id);
}

void draw(View &view, sim::State &state, std::size_t conversation_index,
          voice::Queue &speech, double now)
{
    sim::Conversation &conversation = state.conversations[conversation_index];
    const std::string title = window_name(conversation, conversation_index);
    if (state.conversation_selection_requested &&
        state.selected_conversation == conversation_index) {
        ImGui::SetNextWindowFocus();
    }
    if (!ImGui::Begin(title.c_str(), &conversation.window_open)) {
        ImGui::End();
        return;
    }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        state.selected_conversation = conversation_index;
        state.conversation_selection_requested = false;
    }

    const sim::Instance &instance = state.instances[conversation.instance_index];
    const auto language = std::find_if(state.models.begin(), state.models.end(),
                                       [](const sim::Model &model) {
                                           return model.language_role == "language";
                                       });
    const char *model = language == state.models.end() ? "No language model" : language->name.c_str();
    ImGui::Text("%s. %s. %s.", instance.name.c_str(), conversation.name.c_str(), model);
    if (ImGui::Button("New conversation")) {
        state.create_conversation(conversation.instance_index);
        ImGui::End();
        return;
    }
    ImGui::Separator();

    const float editor_height = ImGui::GetTextLineHeightWithSpacing() * 5.0f;
    const float counter_height = ImGui::GetTextLineHeightWithSpacing() * 1.5f;
    ImGui::BeginChild("Transcript", ImVec2(0.0f, -(editor_height + counter_height)),
                      ImGuiChildFlags_Borders);
    bool continue_requested = false;
    for (const sim::TranscriptEvent &event : conversation.transcript) {
        continue_requested = draw_event(event) || continue_requested;
    }
    if (view.follow && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 8.0f) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    if (continue_requested) {
        state.continue_reply(conversation_index, now);
    }

    view.spoken.resize(conversation.transcript.size(), false);
    for (std::size_t index = 0; index < conversation.transcript.size(); ++index) {
        const sim::TranscriptEvent &event = conversation.transcript[index];
        if (view.spoken[index]) continue;
        if (event.kind == sim::EventKind::tool_call) {
            speech.speak(voice::Source::agent(event.agent_index), event.stated);
            view.spoken[index] = true;
        } else if (event.kind == sim::EventKind::message && !event.streaming &&
                   event.role != sim::Role::user) {
            const voice::Source source = event.role == sim::Role::system
                                             ? voice::Source::system()
                                             : voice::Source::agent(event.agent_index);
            speech.speak(source, event.stated);
            view.spoken[index] = true;
        }
    }

    const bool alternate = ImGui::GetIO().KeyAlt;
    const ImGuiInputTextFlags flags = editor_flags(alternate);
    const bool entered = ImGui::InputTextMultiline("##editor", view.editor.data(),
                                                    view.editor.size(), ImVec2(-1.0f, editor_height),
                                                    flags, editor_callback);
    if (entered && !alternate) {
        state.send(conversation_index, view.editor.data(), now);
        view.editor.fill('\0');
        ImGui::SetKeyboardFocusHere(-1);
    }
    ImGui::TextDisabled("%zu/4000 bytes", std::strlen(view.editor.data()));
    ImGui::SameLine();
    ImGui::TextDisabled("Push Enter to send the line. Push Alt-Enter to start a new line.");
    ImGui::End();
}

void draw(View &view, replica::State &state, std::size_t conversation_index,
          client::Client &socket, voice::Queue &speech)
{
    if (conversation_index >= state.agents().size()) return;
    replica::Agent &agent = state.agents()[conversation_index];
    const std::string title = window_name(agent);
    if (!ImGui::Begin(title.c_str(), &agent.window_open)) {
        ImGui::End();
        return;
    }
    ImGui::Text("%s. %s. %s.", state.journal().string().c_str(),
                agent.conversation.c_str(), state.language_model().c_str());
    ImGui::Separator();

    const float editor_height = ImGui::GetTextLineHeightWithSpacing() * 5.0f;
    const float counter_height = ImGui::GetTextLineHeightWithSpacing() * 1.5f;
    ImGui::BeginChild("Transcript", ImVec2(0.0f, -(editor_height + counter_height)),
                      ImGuiChildFlags_Borders);
    bool continue_requested = false;
    for (std::size_t index = 0u; index < agent.transcript.size(); ++index) {
        const replica::TranscriptEvent &event = agent.transcript[index];
        const bool current_bound = agent.id == 0u && agent.reply_bound &&
                                   index + 1u == agent.transcript.size();
        continue_requested = draw_live_event(event, agent.id, current_bound) ||
                             continue_requested;
    }
    if (view.follow && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 8.0f) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    if (continue_requested) socket.send_line("continue");

    view.spoken.resize(agent.transcript.size(), false);
    if (view.live_agent != agent.id) {
        view.live_agent = agent.id;
        view.spoken.assign(agent.transcript.size(), true);
        for (std::size_t index = 0u; index < agent.transcript.size(); ++index) {
            if (agent.transcript[index].kind == "part") view.spoken[index] = false;
        }
    }
    for (std::size_t index = 0; index < agent.transcript.size(); ++index) {
        const replica::TranscriptEvent &event = agent.transcript[index];
        if (view.spoken[index]) continue;
        if (live_tool(event)) {
            speech.speak(voice::Source::agent(agent.id), stated_tool(agent.id, event));
            view.spoken[index] = true;
        } else if (event.kind == "reply") {
            speech.speak(voice::Source::agent(agent.id), event.text);
            view.spoken[index] = true;
        } else if (live_user(event)) {
            view.spoken[index] = true;
        }
    }

    const bool alternate = ImGui::GetIO().KeyAlt;
    const ImGuiInputTextFlags flags = editor_flags(alternate);
    const bool entered = ImGui::InputTextMultiline("##editor", view.editor.data(),
                                                    view.editor.size(), ImVec2(-1.0f, editor_height),
                                                    flags, editor_callback);
    if (entered && !alternate) {
        const std::string prefix = agent.id == 0u ? "say "
                                                  : "task " + std::to_string(agent.id) + " ";
        if (socket.send_line(prefix + view.editor.data())) view.editor.fill('\0');
        ImGui::SetKeyboardFocusHere(-1);
    }
    ImGui::TextDisabled("%zu/4000 bytes", std::strlen(view.editor.data()));
    ImGui::SameLine();
    ImGui::TextDisabled("Push Enter to send the line. Push Alt-Enter to start a new line.");
    ImGui::End();
}

bool verify_key_paths()
{
    const ImGuiInputTextFlags plain = editor_flags(false);
    const ImGuiInputTextFlags alternate = editor_flags(true);
    return (plain & ImGuiInputTextFlags_EnterReturnsTrue) != 0 &&
           (plain & ImGuiInputTextFlags_CtrlEnterForNewLine) != 0 &&
           (alternate & ImGuiInputTextFlags_EnterReturnsTrue) != 0 &&
           (alternate & ImGuiInputTextFlags_CtrlEnterForNewLine) == 0 &&
           View{}.editor.size() == 4001 && run_key_path(false) && run_key_path(true) &&
           run_under_limit_path() && run_paste_limit_path() && run_callback_limit_path();
}

} // namespace aotx::ctrl::chat
