# The control program uses the vendored ImGui sources with GLFW and OpenGL.
find_package(PkgConfig REQUIRED)
pkg_check_modules(AOTX_CTRL_GLFW REQUIRED glfw3)
find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

set(AOTX_CTRL_VENDOR ${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui)
add_executable(aotx_ctrl
    src/app/app.cpp
    src/browser/browser.cpp
    src/chat/chat.cpp
    src/client/client.cpp
    src/control/control.cpp
    src/instances/instances.cpp
    src/instances/lifecycle.cpp
    src/model/model.cpp
    src/module/module.cpp
    src/monitor/monitor.cpp
    src/monitor/telemetry.cpp
    src/replica/json.cpp
    src/replica/replica.cpp
    src/replica/schema.cpp
    src/replica/store.cpp
    src/settings/settings.cpp
    src/shell/shell.cpp
    src/sim/sim.cpp
    src/theme/theme.cpp
    src/toast/toast.cpp
    src/voice/voice.cpp
    src/wizard/wizard.cpp
    ${AOTX_CTRL_VENDOR}/imgui.cpp
    ${AOTX_CTRL_VENDOR}/imgui_draw.cpp
    ${AOTX_CTRL_VENDOR}/imgui_tables.cpp
    ${AOTX_CTRL_VENDOR}/imgui_widgets.cpp
    ${AOTX_CTRL_VENDOR}/backends/imgui_impl_glfw.cpp
    ${AOTX_CTRL_VENDOR}/backends/imgui_impl_opengl3.cpp)
target_compile_features(aotx_ctrl PRIVATE cxx_std_17)
target_include_directories(aotx_ctrl PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${AOTX_CTRL_GLFW_INCLUDE_DIRS}
    ${AOTX_CTRL_VENDOR}
    ${AOTX_CTRL_VENDOR}/backends)
target_link_libraries(aotx_ctrl PRIVATE
    ${AOTX_CTRL_GLFW_LIBRARIES} OpenGL::GL Threads::Threads ${CMAKE_DL_LIBS})
target_compile_definitions(aotx_ctrl PRIVATE
    AOTX_CTRL_MODEL_CATALOG="${CMAKE_SOURCE_DIR}/share/models/catalog.jsonl")
if(AOTX_PROFILE STREQUAL "8g")
    target_compile_definitions(aotx_ctrl PRIVATE AOTX_CTRL_LANGUAGE_ROLE="language-q4")
else()
    target_compile_definitions(aotx_ctrl PRIVATE AOTX_CTRL_LANGUAGE_ROLE="language")
endif()

add_executable(aotx_ctrl_live_client
    tests/live_client.cpp
    src/client/client.cpp
    src/instances/lifecycle.cpp
    src/monitor/telemetry.cpp
    src/replica/json.cpp
    src/replica/replica.cpp
    src/replica/schema.cpp
    src/replica/store.cpp)
target_compile_features(aotx_ctrl_live_client PRIVATE cxx_std_17)
target_include_directories(aotx_ctrl_live_client PRIVATE
    ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(aotx_ctrl_live_client PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
if(AOTX_PROFILE STREQUAL "8g")
    target_compile_definitions(aotx_ctrl_live_client PRIVATE
                               AOTX_CTRL_LANGUAGE_ROLE="language-q4"
                               AOTX_CTRL_MODEL_CATALOG="${CMAKE_SOURCE_DIR}/share/models/catalog.jsonl")
else()
    target_compile_definitions(aotx_ctrl_live_client PRIVATE
                               AOTX_CTRL_LANGUAGE_ROLE="language"
                               AOTX_CTRL_MODEL_CATALOG="${CMAKE_SOURCE_DIR}/share/models/catalog.jsonl")
endif()

if(AOTX_DISPLAY_TESTS)
    add_test(NAME ctrl_smoke COMMAND aotx_ctrl --sim --frames 300)
    set_tests_properties(ctrl_smoke PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE LABELS display)
endif()
