# The build for aotx. A profile fixes every figure that sizes a device table, and the
# architecture option names the compute capability the device code targets.
cmake_minimum_required(VERSION 3.28)
project(aotx VERSION 0.2.6 LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# The profile of the build. Each name has a header in cuda/profile.
set(AOTX_PROFILE "12g" CACHE STRING "The profile of the build: 8g, 12g, 24g or 48g.")
set_property(CACHE AOTX_PROFILE PROPERTY STRINGS 8g 12g 24g 48g)
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/cuda/profile/${AOTX_PROFILE}.cuh)
    message(FATAL_ERROR "the profile ${AOTX_PROFILE} has no header in cuda/profile")
endif()

# The architecture of the build: 86 for Ampere, 89 for Ada, 120 for Blackwell.
set(AOTX_ARCH "86" CACHE STRING "The compute capability the device code targets.")
set(CMAKE_CUDA_ARCHITECTURES ${AOTX_ARCH})
message(STATUS "aotx profile ${AOTX_PROFILE} arch sm_${AOTX_ARCH}")
# Kernels and device state live in separate files of one module, so the device link runs.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ptxas reports register and spill counts for the spill gate.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

# Every program lands in one directory, so a program finds its siblings beside itself.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_library(AOTX_LIBCUDA cuda REQUIRED)
find_package(Threads REQUIRED)

# A fetch is optional. The store still lists, checks and activates without this library.
find_package(CURL QUIET)
option(AOTX_FETCH "Build the model fetch operation." ${CURL_FOUND})
if(AOTX_FETCH AND NOT CURL_FOUND)
    message(FATAL_ERROR "AOTX_FETCH needs libcurl")
endif()

# The graphical control program is optional and has no effect on the device targets.
option(AOTX_CTRL "Build the graphical control program." ON)

# The affect substrate and its quality instrument are optional together.
option(AOTX_AFFECT "Build the affect substrate and the conversation quality instrument." ON)
if(AOTX_AFFECT)
    add_compile_definitions(AOTX_AFFECT)
endif()

# The modules that run the system. The boot program and the tests link them.
add_library(aotx_core STATIC
    cuda/time/tick.cu
    cuda/profile/card.cu
    cuda/settings/settings.cu
    cuda/settings/settings_host.cu
    cuda/mem/regions.cu
    cuda/mem/tensors.cu
    cuda/mem/mem_host.cu
    cuda/seam/ring.cu
    cuda/seam/flush.cu
    cuda/seam/inbound.cu
    cuda/seam/bulk.cu
    cuda/seam/seam_host.cu
    cuda/bus/append.cu
    cuda/bus/cursor.cu
    cuda/kvcache/pages.cu
    cuda/kvcache/kv_host.cu
    cuda/model/desc.cu
    cuda/model/gather.cu
    cuda/model/norm.cu
    cuda/model/rope.cu
    cuda/model/attention.cu
    cuda/model/conduct.cu
    cuda/model/conduct_host.cu
    cuda/model/sample.cu
    cuda/model/decode.cu
    cuda/model/plan.cu
    cuda/model/commit.cu
    cuda/model/load.cu
    cuda/model/load_host.cu
    cuda/model/layout_host.cu
    cuda/model/product.cu
    cuda/model/desc_host.cu
    cuda/model/decode_host.cu
    cuda/model/graph_host.cu
    cuda/model/model_host.cu
    cuda/model/call_host.cu
    cuda/embed/embed.cu
    cuda/rerank/rerank.cu
    cuda/sched/step.cu
    cuda/sched/pump_host.cu
    cuda/sched/capture_host.cu
    cuda/boot/clock_host.cu
    cuda/boot/card_host.cu
    cuda/boot/options_host.cu
    cuda/boot/phase_host.cu
    cuda/boot/children_host.cu
    cuda/boot/models_host.cu
    cuda/boot/weights_host.cu
    cuda/boot/weights_plan_host.cu
    cuda/text/unicode_tables.cu
    cuda/text/utf8.cu
    cuda/text/fmt.cu
    cuda/text/pretok.cu
    cuda/text/vocab.cu
    cuda/text/bpe.cu
    cuda/text/detok.cu
    cuda/text/vocab_host.cu
    cuda/model/dequant.cu
    cuda/model/gemm.cu
    cuda/model/gemv.cu
    cuda/catalog/catalog.cu
    cuda/catalog/import.cu
    cuda/catalog/manifest.cu
    cuda/catalog/list.cu
    cuda/catalog/console.cu
    cuda/catalog/catalog_host.cu
    cuda/cli/editor.cu
    cuda/cli/parse.cu
    cuda/cli/say.cu
    cuda/cli/say_host.cu
    cuda/agent/table.cu
    cuda/agent/transcript.cu
    cuda/agent/step.cu
    cuda/agent/agent_host.cu
    cuda/tool/parse.cu
    cuda/tool/host_tools.cu
    cuda/tool/device_tools.cu
    cuda/tool/module.cu
    cuda/tool/module_host.cu
    cuda/tool/file_host.cu
    cuda/tool/stream_host.cu
    cuda/tool/node_host.cu
    cuda/tool/tool_host.cu
    cuda/embed/search.cu
)
if(AOTX_AFFECT)
    target_sources(aotx_core PRIVATE cuda/affect/affect.cu cuda/affect/actuate.cu
                                    cuda/affect/readout.cu
                                    cuda/affect/affect_host.cu cuda/affect/calibration_host.cu
                                    cuda/quality/quality.cu cuda/quality/quality_host.cu)
    target_compile_definitions(aotx_core PRIVATE
        AOTX_QUALITY_PHRASE_FILE="${CMAKE_SOURCE_DIR}/tests/fixtures/quality/refusal-phrases.txt")
endif()
# The device tree and the one header a tool module includes. A module compiles with the
# toolkit and this directory alone.
target_include_directories(aotx_core PUBLIC cuda sdk)
target_link_libraries(aotx_core PUBLIC ${AOTX_LIBCUDA})
target_compile_definitions(aotx_core PUBLIC AOTX_PTX_DIR="${CMAKE_SOURCE_DIR}/ptx")
# The module directories the repository carries. The feeder imports every directory below
# the one it is given. The value therefore names the directory that holds the modules and
# not the group above it. The boot program gives the feeder this directory when the command
# line and the settings file name none.
target_compile_definitions(aotx_core
                           PUBLIC AOTX_MODULES_DIR="${CMAKE_SOURCE_DIR}/modules/roles")
target_compile_definitions(aotx_core PUBLIC AOTX_VERSION="${PROJECT_VERSION}")
target_compile_definitions(aotx_core PUBLIC AOTX_ARCH=${AOTX_ARCH})
target_compile_definitions(aotx_core
                           PUBLIC AOTX_PROFILE_HEADER="profile/${AOTX_PROFILE}.cuh")

# The system entry: context, memory map, rings, ticks.
add_executable(aotx_boot cuda/boot/boot_host.cu cuda/boot/signal_host.cu)
target_link_libraries(aotx_boot PRIVATE aotx_core)
set_target_properties(aotx_boot PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(aotx_steer_derive tools/steer_derive.cu tools/steer_derive_host.cu
               tools/steer_calibrate_host.cu)
target_link_libraries(aotx_steer_derive PRIVATE aotx_core)
target_include_directories(aotx_steer_derive PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(aotx_steer_derive PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The module check program. It imports a manifest through the reader of the device. It then
# loads the module of a device tool by its digest and runs it over a batch of the tick. A
# host tool runs through the starter the feeder uses, so the program gets the working
# directory and the line that a run gives it.
add_executable(aotx_module_check
    cuda/catalog/check.cu
    cuda/catalog/check_host.cu
    cuda/catalog/run_host.cu
    cuda/catalog/program_host.cu
    disk/feed/run_tool.c
    disk/feed/fs_tool.c
    disk/feed/fs_path.c
    disk/feed/fs_write.c
    disk/feed/modules.c
    disk/feed/import.c)
target_link_libraries(aotx_module_check PRIVATE aotx_core)
set_target_properties(aotx_module_check PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

enable_testing()
# The clock module check keeps the module path proven at every build.
add_test(NAME clock_module COMMAND aotx_boot --clock-only)
add_test(NAME boot_missing_settings
         COMMAND ${CMAKE_SOURCE_DIR}/tests/boot_missing_test.sh $<TARGET_FILE:aotx_boot>)

# The parity gate holds the action table of the terminal program and the help table of the
# parser together. A screen therefore cannot send a line the device does not parse. The gate
# reads source files alone, so it needs no build and no card. Each arm has a fixture that
# earns a refusal. The fixtures give a line the parser does not name, a key no screen takes,
# and a help line the parser does not dispatch.
add_test(NAME parity COMMAND ${CMAKE_SOURCE_DIR}/tools/parity-gate.py)
add_test(NAME parity_refuse_command
         COMMAND ${CMAKE_SOURCE_DIR}/tools/parity-gate.py --actions
                 ${CMAKE_SOURCE_DIR}/tests/fixtures/parity/actions-bad-command.h)
add_test(NAME parity_refuse_key
         COMMAND ${CMAKE_SOURCE_DIR}/tools/parity-gate.py --actions
                 ${CMAKE_SOURCE_DIR}/tests/fixtures/parity/actions-bad-key.h)
add_test(NAME parity_refuse_help
         COMMAND ${CMAKE_SOURCE_DIR}/tools/parity-gate.py --help-table
                 ${CMAKE_SOURCE_DIR}/tests/fixtures/parity/help-bad.h --actions
                 ${CMAKE_SOURCE_DIR}/tests/fixtures/parity/actions-small.h)
foreach(case command key help)
    set_tests_properties(parity_refuse_${case} PROPERTIES WILL_FAIL TRUE)
endforeach()

add_subdirectory(tests)

# The disk side builds from its own file.
add_subdirectory(disk)

# The model file reader is C on the disk side. The boot glue links it to read a model file
# and to check the record of that file. The settings reader is C on the disk side as well.
# The boot glue reads the settings file with it, and the feeder reads the same file.
target_link_libraries(aotx_core PUBLIC aotx_modelfile aotx_settings)

# The interface builds from its own file; it links the window libraries.
add_subdirectory(cuda/ui)

if(AOTX_CTRL)
    add_subdirectory(ctrl)
endif()
