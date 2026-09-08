// Purpose: Render simulated or live chat events and accept user text.
// Owns: Role cues, tool detail expansion, and editor key actions.
// Launch shape: One chat window processes one editor action per frame.
// Lifetime: Transcript data remains owned by its selected data state.
#include "chat/chat.hpp"

#include "chat/actions.hpp"
#include "chat/render.hpp"
#include "client/client.hpp"
#include "imgui.h"
#include "instances/lifecycle.hpp"
#include "replica/replica.hpp"
#include "theme/theme.hpp"
#include "voice/voice.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
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
    } else if (event.role == sim::Role::agent) {
        draw_formatted(event.stated + (event.streaming ? "|" : ""));
    } else {
        ImGui::TextWrapped("%s", event.stated.c_str());
    }
    if (ImGui::Button("Copy")) ImGui::SetClipboardText(event.stated.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("Copy this message.");
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

std::string simulated_tool(const std::string &stated)
{
    const std::string marker = " calls ";
    const std::size_t first = stated.find(marker);
    if (first == std::string::npos) return "tool";
    const std::size_t name = first + marker.size();
    const std::size_t last = stated.find(' ', name);
    return stated.substr(name, last - name);
}

void draw_confident(const replica::TranscriptEvent &event, unsigned agent,
                    const replica::State &state)
{
    std::vector<const replica::TokenStat *> tokens;
    for (const replica::TokenStat &token : state.tokens()) {
        if (token.agent == agent && token.turn == event.turn) tokens.push_back(&token);
    }
    std::sort(tokens.begin(), tokens.end(), [](const auto *left, const auto *right) {
        return left->index < right->index;
    });
    bool indexed = tokens.size() == event.token_text.size() && !tokens.empty();
    for (std::size_t index = 0u; indexed && index < tokens.size(); ++index) {
        indexed = tokens[index]->index == index;
    }
    if (!indexed) {
        draw_formatted(event.text + (event.kind == "part" ? "|" : ""));
        return;
    }
    for (std::size_t index = 0u; index < tokens.size(); ++index) {
        if (index != 0u) ImGui::SameLine(0.0f, 0.0f);
        const float certainty = static_cast<float>(std::clamp(
            std::exp(tokens[index]->logprob), 0.0, 1.0));
        const ImVec4 low = theme::palette().severity_error;
        const ImVec4 high = ImGui::GetStyleColorVec4(ImGuiCol_Text);
        const ImVec4 color(low.x + (high.x - low.x) * certainty,
                           low.y + (high.y - low.y) * certainty,
                           low.z + (high.z - low.z) * certainty, 1.0f);
        ImGui::TextColored(color, "%s", event.token_text[index].c_str());
    }
}

bool draw_live_event(const replica::TranscriptEvent &event, unsigned agent,
                     bool allow_continue, const replica::State &state, bool confidence)
{
    /* The selection and the completed turn markers are memory records, not conversation. */
    if (event.kind == "selection" ||
        (event.kind == "done" && event.status != "failed" && event.status != "stopped")) {
        return false;
    }
    bool continue_requested = false;
    ImGui::PushID(&event);
    const sim::Role role = live_user(event) ? sim::Role::user : sim::Role::agent;
    ImGui::TextColored(role_color(role), "%s", role_name(role));
    std::string copy = event.text;
    if (event.kind == "done" && event.status == "stopped") {
        ImGui::TextDisabled("The reply was stopped.");
        copy = "The reply was stopped.";
    } else if (event.kind == "done") {
        ImGui::TextDisabled("The turn did not complete.");
        copy = "The turn did not complete.";
    } else if (live_tool(event)) {
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
        if (live_user(event)) ImGui::TextWrapped("%s", event.text.c_str());
        else if (confidence) draw_confident(event, agent, state);
        else draw_formatted(event.text + (event.kind == "part" ? "|" : ""));
    } else {
        ImGui::TextWrapped("%s request %llu, %s", event.tool.c_str(),
                           static_cast<unsigned long long>(event.request), event.status.c_str());
    }
    if (ImGui::Button("Copy")) ImGui::SetClipboardText(copy.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("Copy this message.");
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

template <std::size_t Size>
void set_text(std::array<char, Size> &out, const std::string &text)
{
    std::strncpy(out.data(), text.c_str(), out.size() - 1u);
    out.back() = '\0';
}

void load_persona(View &view, const replica::State &state, const replica::Agent &agent,
                  persona::Store &personas)
{
    const std::string key = state.journal().string() + ":" + std::to_string(agent.id);
    if (view.persona_loaded && view.persona_key == key) return;
    view.persona_loaded = true;
    view.persona_key = key;
    set_text(view.default_voice, personas.default_voice(state.journal()));
    std::string over;
    view.override_on = personas.override_voice(state.journal(), agent.id, over);
    set_text(view.override_voice, over);
    set_text(view.conversation_name, agent.conversation);
    view.action_result.clear();
}

void draw_persona(View &view, const replica::State &state, const replica::Agent &agent,
                  persona::Store &personas)
{
    if (!ImGui::CollapsingHeader("Persona")) return;
    ImGui::TextWrapped("%s", persona::identity_spine().c_str());
    ImGui::InputTextMultiline("Default voice", view.default_voice.data(),
                              view.default_voice.size(), ImVec2(-1.0f, 70.0f));
    if (ImGui::Button("Save##default-persona")) {
        personas.save_default(state.journal(), view.default_voice.data(), view.action_result);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Save the default voice for this instance.");
    ImGui::Checkbox("Override", &view.override_on);
    ImGui::SameLine();
    ImGui::TextDisabled("Use a voice that belongs to this conversation.");
    if (!view.override_on) ImGui::BeginDisabled();
    ImGui::InputTextMultiline("Voice", view.override_voice.data(),
                              view.override_voice.size(), ImVec2(-1.0f, 70.0f));
    if (!view.override_on) ImGui::EndDisabled();
    if (ImGui::Button("Save##conversation-persona")) {
        const char *voice = view.override_on ? view.override_voice.data() : "";
        personas.save_override(state.journal(), agent.id, voice, view.action_result);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Save or disable this conversation voice.");
    ImGui::TextWrapped("%s", persona::conduct_floor().c_str());
    const char *voice = view.override_on ? view.override_voice.data() : view.default_voice.data();
    if (ImGui::TreeNode("Composed text")) {
        ImGui::TextWrapped("%s", persona::compose(voice).c_str());
        ImGui::TreePop();
    }
    ImGui::TextDisabled("A new conversation imports this text as a role overlay.");
}

void advance_persona(View &view, const replica::State &state, client::Client &socket)
{
    if (view.pending_role.empty()) return;
    const std::string installed = "import: the role " + view.pending_role + " is installed";
    const std::string spawn = "spawn: " + view.pending_role + " on slots ";
    for (auto at = state.notes().rbegin(); at != state.notes().rend(); ++at) {
        if (view.persona_spawning && at->text.rfind(spawn, 0u) == 0u) {
            view.action_result = "The persona role " + view.pending_role +
                                 " was imported and spawned.";
            view.pending_role.clear();
            view.persona_spawning = false;
            return;
        }
        if (view.persona_importing && at->text == installed) {
            if (socket.send_line("spawn " + view.pending_role)) {
                view.action_result = "The persona role " + view.pending_role +
                                     " was imported. Its spawn request was sent.";
                view.persona_importing = false;
                view.persona_spawning = true;
            }
            return;
        }
        if (view.persona_importing && at->text.rfind("import: the role " +
            view.pending_role + " was refused", 0u) == 0u) {
            view.action_result = at->text;
            view.pending_role.clear();
            view.persona_importing = false;
            return;
        }
    }
}

void new_conversation(View &view, replica::State &state, client::Client &socket)
{
    const char *selected = view.override_on ? view.override_voice.data()
                                            : view.default_voice.data();
    if (selected[0] == '\0') {
        std::string command;
        if (state.create_conversation(command)) {
            socket.send_line(command);
        }
        return;
    }
    persona::RoleModule module;
    if (!persona::write_role_module(state.journal().parent_path() / "persona-modules",
                                    selected, module, view.action_result)) return;
    if (socket.send_line("import " + module.directory.string())) {
        view.pending_role = module.name;
        view.persona_importing = true;
        view.persona_spawning = false;
        view.action_result = "The persona role import request was sent.";
    }
}

void draw_ratio(const replica::State &state, unsigned agent)
{
    unsigned turn = 0u;
    for (const replica::TokenStat &token : state.tokens()) {
        if (token.agent == agent) turn = std::max(turn, token.turn);
    }
    unsigned think = 0u;
    unsigned output = 0u;
    for (const replica::TokenStat &token : state.tokens()) {
        if (token.agent != agent || token.turn != turn) continue;
        if (token.think) ++think;
        else ++output;
    }
    if (think + output == 0u) {
        ImGui::TextDisabled("Think-to-output ratio is not available for this turn.");
    } else if (output == 0u) {
        ImGui::Text("Think-to-output ratio has %u think tokens and no output token.", think);
    } else {
        ImGui::Text("Think-to-output ratio %.2f (%u to %u tokens).",
                    static_cast<double>(think) / output, think, output);
    }
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
    if (ImGui::Button("New")) {
        state.create_conversation(conversation.instance_index);
        ImGui::End();
        return;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Start a new conversation.");
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
            speech.speak(voice::Category::tool, voice::Source::agent(event.agent_index),
                         voice::tool_line(event.agent_index, simulated_tool(event.stated)));
            view.spoken[index] = true;
        } else if (event.kind == sim::EventKind::message && !event.streaming &&
                   event.role != sim::Role::user) {
            const voice::Source source = event.role == sim::Role::system
                                             ? voice::Source::system()
                                             : voice::Source::agent(event.agent_index);
            const voice::Category category = event.role == sim::Role::system
                ? voice::Category::lifecycle : voice::Category::reply;
            speech.speak(category, source, event.stated);
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
          client::Client &socket, voice::Queue &speech, instances::Lifecycle &lifecycle,
          std::size_t instance, persona::Store &personas, bool confidence)
{
    if (conversation_index >= state.agents().size()) return;
    replica::Agent &agent = state.agents()[conversation_index];
    load_persona(view, state, agent, personas);
    advance_persona(view, state, socket);
    const std::string title = window_name(agent);
    ImGui::SetNextWindowSize(ImVec2(720.0f, 680.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(title.c_str(), &agent.window_open)) {
        ImGui::End();
        return;
    }
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        state.select_agent(conversation_index);
    }
    ImGui::Text("%s. %s. %s.", state.journal().string().c_str(),
                agent.conversation.c_str(), state.language_model().c_str());
    draw_ratio(state, agent.id);
    ImGui::SetNextItemWidth(220.0f);
    ImGui::InputText("Name", view.conversation_name.data(), view.conversation_name.size());
    ImGui::SameLine();
    if (ImGui::Button("Save##conversation-name")) {
        if (lifecycle.name_conversation(instance, agent.id, view.conversation_name.data())) {
            view.action_result = "The conversation name was saved.";
        } else {
            view.action_result = lifecycle.refusal();
        }
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Save this conversation name in the instance registry.");
    ImGui::BeginDisabled(!view.pending_role.empty());
    if (ImGui::Button("New")) new_conversation(view, state, socket);
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("Start a new worker conversation.");
    if (ImGui::Button("Export")) {
        std::filesystem::path written;
        export_conversation(state.journal().parent_path() / "exports", agent, written,
                            view.action_result);
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Write this conversation to a text file.");
    if (!agent.reply_in_flight) ImGui::BeginDisabled();
    if (ImGui::Button("Stop") && agent.reply_in_flight) {
        if (socket.send_line(stop_command(agent.id))) {
            view.action_result = "The stop request was sent.";
        }
    }
    if (!agent.reply_in_flight) ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("Stop the reply at its next token.");
    draw_persona(view, state, agent, personas);
    if (!view.action_result.empty()) ImGui::TextWrapped("%s", view.action_result.c_str());
    ImGui::Separator();

    const float editor_height = ImGui::GetTextLineHeightWithSpacing() * 5.0f;
    const float counter_height = ImGui::GetTextLineHeightWithSpacing() * 1.5f;
    ImGui::BeginChild("Transcript", ImVec2(0.0f, -(editor_height + counter_height)),
                      ImGuiChildFlags_Borders);
    bool continue_requested = false;
    for (std::size_t index = 0u; index < agent.transcript.size(); ++index) {
        const replica::TranscriptEvent &event = agent.transcript[index];
        const bool current_bound = agent.reply_bound &&
                                   index + 1u == agent.transcript.size();
        continue_requested = draw_live_event(event, agent.id, current_bound, state, confidence) ||
                             continue_requested;
    }
    if (view.follow && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 8.0f) {
        ImGui::SetScrollHereY(1.0f);
    }
    ImGui::EndChild();
    // The conductor takes the plain command; a worker takes the agent form.
    if (continue_requested) {
        socket.send_line(agent.id == 0u ? std::string("continue")
                                        : "agent " + std::to_string(agent.id) + " continue");
    }

    /* A fresh or rebound window treats the whole transcript as history: shown, not spoken. */
    if (!view.bound || view.live_agent != agent.id ||
        view.spoken.size() > agent.transcript.size()) {
        view.bound = true;
        view.live_agent = agent.id;
        view.spoken.assign(agent.transcript.size(), true);
        for (std::size_t index = 0u; index < agent.transcript.size(); ++index) {
            if (agent.transcript[index].kind == "part") view.spoken[index] = false;
        }
    }
    view.spoken.resize(agent.transcript.size(), false);
    for (std::size_t index = 0; index < agent.transcript.size(); ++index) {
        const replica::TranscriptEvent &event = agent.transcript[index];
        if (view.spoken[index]) continue;
        if (live_tool(event)) {
            speech.speak(voice::Category::tool, voice::Source::agent(agent.id),
                         voice::tool_line(agent.id, event.tool));
            view.spoken[index] = true;
        } else if (event.kind == "reply") {
            speech.speak(voice::Category::reply, voice::Source::agent(agent.id), event.text);
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
