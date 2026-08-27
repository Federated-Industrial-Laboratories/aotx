# The interface: panels, raster, window. Targets and tests for the interface only.
find_package(PkgConfig REQUIRED)
pkg_check_modules(AOTX_GLFW REQUIRED glfw3)
pkg_check_modules(AOTX_GLEW REQUIRED glew)
find_package(OpenGL REQUIRED)
# The window check sends the close request of a window manager on its own connection.
find_package(X11 REQUIRED)

# The panels and the raster. The command layer builds with the core library, so this
# library holds the display alone and reads the state of every other module.
add_library(aotx_ui STATIC
    font.cu
    grid.cu
    panels.cu
    status.cu
    raster.cu
    raster_host.cu)
target_link_libraries(aotx_ui PUBLIC aotx_core)

# The window and the copy into the pixel buffer object.
add_library(aotx_window STATIC window_host.cu)
target_link_libraries(aotx_window PUBLIC aotx_ui ${AOTX_GLFW_LIBRARIES}
                      ${AOTX_GLEW_LIBRARIES} OpenGL::GL)
target_include_directories(aotx_window PUBLIC ${AOTX_GLFW_INCLUDE_DIRS}
                           ${AOTX_GLEW_INCLUDE_DIRS})

# The boot program takes the window option, so it takes the window wiring and the window.
target_sources(aotx_boot PRIVATE ${CMAKE_SOURCE_DIR}/cuda/boot/window_boot_host.cu)
target_link_libraries(aotx_boot PRIVATE aotx_window Threads::Threads)

# The command line check and the panel check run with no window.
foreach(name cli ui)
    add_executable(aotx_${name}_test ${CMAKE_SOURCE_DIR}/tests/${name}_test.cu)
    target_link_libraries(aotx_${name}_test PRIVATE aotx_ui Threads::Threads)
    set_target_properties(aotx_${name}_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME ${name} COMMAND aotx_${name}_test)
endforeach()

# The raster check with no window. The context comes from a display device, which needs no
# display and no desktop, so the default check never opens a window.
pkg_check_modules(AOTX_EGL REQUIRED egl)
add_executable(aotx_raster_headless_test
               ${CMAKE_SOURCE_DIR}/tests/raster_headless_test.cu)
target_link_libraries(aotx_raster_headless_test PRIVATE aotx_ui ${AOTX_EGL_LIBRARIES}
                      ${AOTX_GLEW_LIBRARIES} OpenGL::GL Threads::Threads)
target_include_directories(aotx_raster_headless_test PRIVATE ${AOTX_EGL_INCLUDE_DIRS}
                           ${AOTX_GLEW_INCLUDE_DIRS})
set_target_properties(aotx_raster_headless_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME raster_headless COMMAND aotx_raster_headless_test)
# The check holds the pixel path of the device, so it never runs beside another check.
set_tests_properties(raster_headless PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)

# The check that opens a window on the display of the operator runs only when it is asked
# for. It carries the label display, and it is the tool that sends a close request.
option(AOTX_DISPLAY_TESTS "register the checks that open a window on the display" OFF)
if(AOTX_DISPLAY_TESTS)
    add_executable(aotx_window_test ${CMAKE_SOURCE_DIR}/tests/window_test.cu)
    target_link_libraries(aotx_window_test PRIVATE aotx_window Threads::Threads
                          ${X11_LIBRARIES})
    set_target_properties(aotx_window_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME window COMMAND aotx_window_test ${CMAKE_BINARY_DIR}/window.ppm)
    # The window check holds the display, so it never runs beside another check.
    set_tests_properties(window PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE LABELS display)
endif()
