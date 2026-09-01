// Purpose: Apply the control program palette to ImGui.
// Owns: The one palette instance and all style values.
// Launch shape: One user interface thread applies the style once.
// Lifetime: The style remains active until ImGui stops.
#include "theme/theme.hpp"

namespace aotx::ctrl::theme {
namespace {

ImVec4 color(unsigned value, float alpha = 1.0f)
{
    return ImVec4(((value >> 16) & 0xff) / 255.0f,
                  ((value >> 8) & 0xff) / 255.0f,
                  (value & 0xff) / 255.0f, alpha);
}

const Palette colors = {
    color(0x51b37f), color(0xd6a64f), color(0x737985),
    color(0x77a9d6), color(0xb0b5bd), color(0xa58bd4),
    color(0x77a9d6), color(0x51b37f), color(0xd6a64f), color(0xd66b6b)};

} // namespace

const Palette &palette()
{
    return colors;
}

void apply()
{
    ImGuiStyle &style = ImGui::GetStyle();
    ImVec4 *values = style.Colors;
    const ImVec4 panel = color(0x0a0c0f);
    const ImVec4 input = color(0x06080a);
    const ImVec4 raised = color(0x12161b);
    const ImVec4 hover = color(0x1a2027);
    const ImVec4 text = color(0xe2e5e9);
    const ImVec4 muted = color(0x7a828d);
    const ImVec4 border = color(0x303741);
    const ImVec4 accent = color(0x77a9d6);
    const ImVec4 clear(0.0f, 0.0f, 0.0f, 0.0f);

    values[ImGuiCol_Text] = text;
    values[ImGuiCol_TextDisabled] = muted;
    values[ImGuiCol_WindowBg] = panel;
    values[ImGuiCol_ChildBg] = panel;
    values[ImGuiCol_PopupBg] = raised;
    values[ImGuiCol_Border] = border;
    values[ImGuiCol_BorderShadow] = clear;
    values[ImGuiCol_FrameBg] = input;
    values[ImGuiCol_FrameBgHovered] = hover;
    values[ImGuiCol_FrameBgActive] = raised;
    values[ImGuiCol_TitleBg] = raised;
    values[ImGuiCol_TitleBgActive] = hover;
    values[ImGuiCol_TitleBgCollapsed] = raised;
    values[ImGuiCol_MenuBarBg] = raised;
    values[ImGuiCol_ScrollbarBg] = input;
    values[ImGuiCol_ScrollbarGrab] = border;
    values[ImGuiCol_ScrollbarGrabHovered] = muted;
    values[ImGuiCol_ScrollbarGrabActive] = accent;
    values[ImGuiCol_CheckMark] = accent;
    values[ImGuiCol_SliderGrab] = accent;
    values[ImGuiCol_SliderGrabActive] = accent;
    values[ImGuiCol_Button] = clear;
    values[ImGuiCol_ButtonHovered] = hover;
    values[ImGuiCol_ButtonActive] = raised;
    values[ImGuiCol_Header] = clear;
    values[ImGuiCol_HeaderHovered] = hover;
    values[ImGuiCol_HeaderActive] = raised;
    values[ImGuiCol_Separator] = border;
    values[ImGuiCol_SeparatorHovered] = muted;
    values[ImGuiCol_SeparatorActive] = accent;
    values[ImGuiCol_ResizeGrip] = clear;
    values[ImGuiCol_ResizeGripHovered] = border;
    values[ImGuiCol_ResizeGripActive] = accent;
    values[ImGuiCol_Tab] = raised;
    values[ImGuiCol_TabHovered] = hover;
    values[ImGuiCol_TabSelected] = hover;
    values[ImGuiCol_TabDimmed] = raised;
    values[ImGuiCol_TabDimmedSelected] = hover;
    values[ImGuiCol_TabSelectedOverline] = accent;
    values[ImGuiCol_DockingPreview] = ImVec4(accent.x, accent.y, accent.z, 0.7f);
    values[ImGuiCol_DockingEmptyBg] = panel;
    values[ImGuiCol_PlotHistogram] = accent;
    values[ImGuiCol_TextSelectedBg] = ImVec4(accent.x, accent.y, accent.z, 0.25f);
    values[ImGuiCol_NavCursor] = accent;
    values[ImGuiCol_DragDropTarget] = accent;

    style.WindowRounding = 0.0f;
    style.ChildRounding = 0.0f;
    style.FrameRounding = 0.0f;
    style.PopupRounding = 0.0f;
    style.ScrollbarRounding = 0.0f;
    style.TabRounding = 0.0f;
    style.GrabRounding = 0.0f;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.FrameBorderSize = 1.0f;
    style.ScrollbarSize = 8.0f;
    style.WindowPadding = ImVec2(12.0f, 12.0f);
    style.FramePadding = ImVec2(8.0f, 5.0f);
    style.ItemSpacing = ImVec2(8.0f, 7.0f);
}

} // namespace aotx::ctrl::theme
