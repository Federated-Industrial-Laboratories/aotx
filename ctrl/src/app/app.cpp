// Purpose: Run GLFW, OpenGL, ImGui, and all control shell modules.
// Owns: Window resources, ImGui resources, and the frame loop.
// Launch shape: One process draws one frame at a time.
// Lifetime: Resources start in order and stop in reverse order.
#include "app/app.hpp"

#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "chat/chat.hpp"
#include "imgui.h"
#include "instances/instances.hpp"
#include "shell/shell.hpp"
#include "sim/sim.hpp"
#include "theme/theme.hpp"
#include "toast/toast.hpp"

#include <GLFW/glfw3.h>

#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace aotx::ctrl::app {
namespace {

struct Options {
    int frame_limit = -1;
};

void glfw_error(int, const char *description)
{
    std::fprintf(stderr, "GLFW reports: %s\n", description);
}

bool parse_options(int argc, char **argv, Options &options)
{
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument != "--frames" || index + 1 >= argc) {
            std::fputs("AOTX-CTRL refuses an unknown option.\n", stderr);
            return false;
        }
        const std::string value = argv[++index];
        int parsed = -1;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (result.ec != std::errc() || result.ptr != value.data() + value.size() || parsed < 0) {
            std::fputs("AOTX-CTRL refuses an invalid frame count.\n", stderr);
            return false;
        }
        options.frame_limit = parsed;
    }
    return true;
}

bool make_layout_path(std::string &path)
{
    const char *xdg = std::getenv("XDG_CONFIG_HOME");
    const char *home = std::getenv("HOME");
    std::filesystem::path directory;
    if (xdg != nullptr && xdg[0] != '\0') {
        directory = xdg;
    } else if (home != nullptr && home[0] != '\0') {
        directory = std::filesystem::path(home) / ".config";
    } else {
        std::fputs("AOTX-CTRL cannot find the user configuration directory.\n", stderr);
        return false;
    }
    directory /= "aotx";
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) {
        std::fputs("AOTX-CTRL cannot create the user configuration directory.\n", stderr);
        return false;
    }
    path = (directory / "ctrl-layout.ini").string();
    return true;
}

int run_loop(GLFWwindow *window, int frame_limit, const std::string &layout_path)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    io.IniFilename = layout_path.c_str();
    theme::apply();

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true) || !ImGui_ImplOpenGL3_Init("#version 130")) {
        std::fputs("AOTX-CTRL cannot start the graphics interface.\n", stderr);
        ImGui::DestroyContext();
        return 1;
    }

    sim::State simulated;
    shell::State shell_state;
    chat::View chat_view;
    toast::Lane toasts;
    int frames = 0;
    while (!glfwWindowShouldClose(window) && (frame_limit < 0 || frames < frame_limit)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        const double now = glfwGetTime();
        simulated.tick(now);
        shell::draw_dock_space(shell_state, simulated);
        if (shell_state.show_instances) {
            instances::draw(simulated, toasts, now, &shell_state.show_instances);
        }
        if (shell_state.show_chat) {
            chat::draw(chat_view, simulated, now, &shell_state.show_chat);
        }
        toasts.draw(now);

        ImGui::Render();
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.035f, 0.043f, 0.052f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        if ((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0) {
            GLFWwindow *context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(context);
        }
        glfwSwapBuffers(window);
        ++frames;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    return 0;
}

} // namespace

int run(int argc, char **argv)
{
    Options options;
    std::string layout_path;
    if (!parse_options(argc, argv, options) || !make_layout_path(layout_path)) {
        return 2;
    }

    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        std::fputs("AOTX-CTRL cannot start GLFW.\n", stderr);
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow *window = glfwCreateWindow(1280, 800, "AOTX-CTRL", nullptr, nullptr);
    if (window == nullptr) {
        std::fputs("AOTX-CTRL cannot open its main window.\n", stderr);
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    const int result = run_loop(window, options.frame_limit, layout_path);
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}

} // namespace aotx::ctrl::app

int main(int argc, char **argv)
{
    return aotx::ctrl::app::run(argc, argv);
}
