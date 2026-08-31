# The control program uses the vendored ImGui sources with GLFW and OpenGL.
find_package(PkgConfig REQUIRED)
pkg_check_modules(AOTX_CTRL_GLFW REQUIRED glfw3)
find_package(OpenGL REQUIRED)

set(AOTX_CTRL_VENDOR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui)
add_executable(aotx_ctrl
    src/app/app.cpp
    src/chat/chat.cpp
    src/instances/instances.cpp
    src/shell/shell.cpp
    src/sim/sim.cpp
    src/theme/theme.cpp
    src/toast/toast.cpp
    ${AOTX_CTRL_VENDOR}/imgui.cpp
    ${AOTX_CTRL_VENDOR}/imgui_draw.cpp
    ${AOTX_CTRL_VENDOR}/imgui_tables.cpp
    ${AOTX_CTRL_VENDOR}/imgui_widgets.cpp
    ${AOTX_CTRL_VENDOR}/backends/imgui_impl_glfw.cpp
    ${AOTX_CTRL_VENDOR}/backends/imgui_impl_opengl3.cpp)
target_compile_features(aotx_ctrl PRIVATE cxx_std_17)
target_include_directories(aotx_ctrl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${AOTX_CTRL_GLFW_INCLUDE_DIRS}
    ${AOTX_CTRL_VENDOR}
    ${AOTX_CTRL_VENDOR}/backends)
target_link_libraries(aotx_ctrl PRIVATE ${AOTX_CTRL_GLFW_LIBRARIES} OpenGL::GL)

if(AOTX_DISPLAY_TESTS)
    add_test(NAME ctrl_smoke COMMAND aotx_ctrl --frames 300)
    set_tests_properties(ctrl_smoke PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE LABELS display)
endif()
