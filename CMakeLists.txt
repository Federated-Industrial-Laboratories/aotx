# The build for aotx. A profile fixes every figure that sizes a device table, and the
# architecture option names the compute capability the device code targets.
cmake_minimum_required(VERSION 3.28)
project(aotx VERSION 0.3.0 LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# The profile of the build. Each name has a header in cuda/profile.
set(AOTX_PROFILE "12g" CACHE STRING "The profile of the build: 8g, 12g, 24g or 48g.")
set_property(CACHE AOTX_PROFILE PROPERTY STRINGS 8g 12g 24g 48g)
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/cuda/profile/${AOTX_PROFILE}.cuh)
    message(FATAL_ERROR "the profile ${AOTX_PROFILE} has no header in cuda/profile")
endif()

# Memory capacity is independent of the stored object schema.
set(AOTX_MEMORY_OBJECTS "8192" CACHE STRING "The number of resident memory object versions.")
set(AOTX_MEMORY_BYTES "16777216" CACHE STRING "The resident memory payload capacity in bytes.")
set(AOTX_CCIR_FILE_BYTES "17179869184" CACHE STRING "The CCIR file byte cap; zero uses the system offset limit.")
string(LENGTH "${AOTX_CCIR_FILE_BYTES}" AOTX_CCIR_FILE_DIGITS)
if(NOT AOTX_CCIR_FILE_BYTES MATCHES "^(0|[1-9][0-9]*)$" OR
   AOTX_CCIR_FILE_DIGITS GREATER 19 OR
   (AOTX_CCIR_FILE_DIGITS EQUAL 19 AND AOTX_CCIR_FILE_BYTES STRGREATER "9223372036854775807") OR
   (AOTX_CCIR_FILE_BYTES GREATER 0 AND AOTX_CCIR_FILE_BYTES LESS 12288))
    message(FATAL_ERROR "CCIR file capacity is outside the file offset range")
endif()
set(AOTX_MEMORY_SNAPSHOTS "2" CACHE STRING "The number of pending memory checkpoint slots.")
if(NOT AOTX_MEMORY_SNAPSHOTS MATCHES "^[1-9][0-9]*$" OR AOTX_MEMORY_SNAPSHOTS GREATER 65536)
    message(FATAL_ERROR "checkpoint ring count is outside the transport range")
endif()
foreach(value AOTX_MEMORY_OBJECTS AOTX_MEMORY_BYTES)
    if(NOT "${${value}}" MATCHES "^[1-9][0-9]*$")
        message(FATAL_ERROR "${value} must be a positive decimal integer within the transfer limit")
    endif()
    string(LENGTH "${${value}}" digits)
    if(digits GREATER 10)
        message(FATAL_ERROR "${value} exceeds the transfer limit")
    endif()
endforeach()
math(EXPR AOTX_MEMORY_IMAGE "128 + 256 * ${AOTX_MEMORY_OBJECTS} + ${AOTX_MEMORY_BYTES}")
if(AOTX_MEMORY_IMAGE GREATER 2147483639)
    message(FATAL_ERROR "memory capacity exceeds the live transfer byte count")
endif()
add_library(aotx_memory_config INTERFACE)
target_compile_definitions(aotx_memory_config INTERFACE
    AOTX_MEMORY_OBJECTS=${AOTX_MEMORY_OBJECTS}u AOTX_MEMORY_BYTES=${AOTX_MEMORY_BYTES}u
    AOTX_MEMORY_SNAPSHOTS=${AOTX_MEMORY_SNAPSHOTS}u AOTX_PROFILE_HEADER="profile/${AOTX_PROFILE}.cuh")
link_libraries(aotx_memory_config)
message(STATUS "memory objects ${AOTX_MEMORY_OBJECTS} payload bytes ${AOTX_MEMORY_BYTES}")

# The architecture of the build: 86 for Ampere, 89 for Ada, 120 for Blackwell.
set(AOTX_ARCH "86" CACHE STRING "The compute capability the device code targets.")
set(CMAKE_CUDA_ARCHITECTURES ${AOTX_ARCH})
message(STATUS "aotx profile ${AOTX_PROFILE} arch sm_${AOTX_ARCH}")
# Kernels and device state live in separate files of one module, so the device link runs.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# The 48g device tables produce host symbols larger than the small address model permits.
if(AOTX_PROFILE STREQUAL "48g" AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$"
   AND CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mcmodel=medium>)
endif()

# ptxas reports register and spill counts for the spill gate.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)

# Every program lands in one directory, so a program finds its siblings beside itself.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_library(AOTX_LIBCUDA cuda REQUIRED)
find_package(CUDAToolkit REQUIRED)
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
include(cuda/profile/config.cmake)
include(cuda/kvcache/config.cmake)
include(cuda/media/config.cmake)
include(cuda/service/config.cmake)
include(cuda/shared/config.cmake)
include(cuda/policy/config.cmake)
add_subdirectory(cuda/media)
add_subdirectory(cuda/vision)
include(cuda/audio/config.cmake)
add_subdirectory(cuda/audio)
add_library(aotx_core STATIC
    cuda/policy/work.cu cuda/policy/rules.cu cuda/policy/replay.cu
    cuda/policy/command.cu cuda/policy/module_host.cu
    cuda/tool/policy.cu
    cuda/tool/policy_command.cu
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
    cuda/model/qk_norm.cu
    cuda/model/experts.cu
    cuda/model/expert_matrix.cu
    cuda/model/experts_host.cu
    cuda/model/bias_host.cu
    cuda/model/delta.cu
    cuda/model/delta_host.cu
    cuda/model/gated.cu
    cuda/model/gated_host.cu
    cuda/model/hybrid.cu
    cuda/model/hybrid_host.cu
    cuda/model/hybrid_file_host.cu
    cuda/model/kinds_host.cu
    cuda/model/attention.cu
    cuda/model/conduct.cu
    cuda/model/control_host.cu
    cuda/model/control.cu
    cuda/model/qualification.cu
    cuda/model/selection.cu
    cuda/model/qualification_host.cu
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
    cuda/model/desc_file_host.cu
    cuda/model/wrap.cu
    cuda/model/wrap_host.cu
    cuda/model/call_format.cu
    cuda/model/decode_module_host.cu
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
    cuda/boot/runtime_host.cu
    cuda/boot/replay_host.cu
    cuda/boot/phase_host.cu
    cuda/boot/children_host.cu
    cuda/boot/models_host.cu
    cuda/boot/vocab_host.cu
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
    cuda/text/vocab_select.cu
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
    cuda/tool/outcome.cu
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
               tools/steer_calibrate_host.cu tools/steer_measure.cu tools/steer_measure_host.cu)
target_link_libraries(aotx_steer_derive PRIVATE aotx_core)
target_include_directories(aotx_steer_derive PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(aotx_steer_derive PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The score tool shares the text sets, the passes and the pack kernel of the steer tool.
add_executable(aotx_quality_score tools/quality_score.cu tools/quality_score_host.cu
               tools/quality_pair_host.cu tools/steer_derive.cu tools/steer_measure.cu tools/steer_measure_host.cu)
target_link_libraries(aotx_quality_score PRIVATE aotx_core)
target_include_directories(aotx_quality_score PRIVATE ${CMAKE_SOURCE_DIR})
set_target_properties(aotx_quality_score PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

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

# Typed state also serves the optional live memory nodes.
add_library(aotx_cognitive STATIC cuda/cognitive/state.cu cuda/cognitive/resolve.cu
    cuda/cognitive/recall.cu cuda/cognitive/recall_record.cu cuda/cognitive/recall_replay.cu)
target_include_directories(aotx_cognitive PUBLIC cuda)
target_link_libraries(aotx_cognitive PUBLIC aotx_memory_config)
target_sources(aotx_core PRIVATE cuda/cognitive/live_transfer.cu cuda/cognitive/live_admission.cu cuda/cognitive/live_stage.cu
    cuda/cognitive/cold.cu cuda/cognitive/cold_result.cu
    cuda/cognitive/live_choice.cu cuda/cognitive/live_prompt.cu cuda/cognitive/live_text.cu
    cuda/cognitive/intake_model.cu cuda/cognitive/intake_index.cu cuda/cognitive/intake_grammar.cu
    cuda/cognitive/source_spans.cu cuda/cognitive/source_table.cu cuda/cognitive/intake_consumed.cu
    cuda/cognitive/intake_capability.cu
    cuda/cognitive/maintenance.cu
    cuda/cognitive/maintenance_request.cu
    cuda/cognitive/checkpoint.cu cuda/cognitive/checkpoint_encode.cu
    cuda/cognitive/checkpoint_import.cu cuda/cognitive/checkpoint_host.cu)
target_sources(aotx_core PRIVATE cuda/appraisal/state.cu cuda/appraisal/queue.cu
    cuda/appraisal/request.cu cuda/appraisal/work.cu cuda/appraisal/control.cu
    cuda/appraisal/command.cu cuda/appraisal/model.cu)
target_sources(aotx_core PRIVATE cuda/media/registry.cu cuda/media/work.cu cuda/media/ingest.cu cuda/media/prompt.cu
    cuda/media/runtime_host.cu cuda/media/load_host.cu cuda/media/ring_host.cu
    cuda/audio/runtime_host.cu cuda/audio/load_host.cu cuda/audio/work.cu)
target_link_libraries(aotx_core PUBLIC aotx_cognitive aotx_image aotx_vision aotx_audio)
target_sources(aotx_core PRIVATE cuda/service/admission.cu cuda/service/grants.cu
    cuda/service/prompt.cu cuda/service/work.cu cuda/service/information.cu cuda/service/media.cu
    cuda/service/media_receipt.cu
    cuda/service/ring_host.cu cuda/boot/service_host.cu)
target_sources(aotx_core PRIVATE cuda/shared/state.cu cuda/shared/admission.cu
    cuda/shared/apply.cu cuda/shared/records.cu cuda/shared/transitions.cu
    cuda/shared/read.cu cuda/shared/bridge_host.cu cuda/shared/bridge_input.cu
    cuda/shared/bridge_memory.cu cuda/shared/bridge_model.cu cuda/shared/bridge_work.cu
    cuda/shared/memory_read.cu cuda/shared/memory_publish.cu cuda/shared/affect.cu)
add_executable(aotx_service disk/service/broker.c disk/service/files.c)
target_include_directories(aotx_service PRIVATE ${CMAKE_SOURCE_DIR} cuda)
target_compile_options(aotx_service PRIVATE -Wall -Wextra -Werror)
add_library(aotx_cognitive_disk STATIC disk/cognitive/io.c disk/cognitive/checkpoint_read.c
    disk/cognitive/cold_read.c
    disk/cognitive/checkpoint_write.c disk/cognitive/checkpoint_drain.c
    disk/runtime/checkpoint.c disk/runtime/replay_write.c disk/runtime/activate.c disk/restore/scan.c)
target_include_directories(aotx_cognitive_disk PUBLIC disk cuda)
target_link_libraries(aotx_cognitive_disk PUBLIC aotx_ccir_store aotx_memory_config aotx_modelfile)
target_link_libraries(aotx_cognitive_disk PUBLIC pthread)
target_sources(aotx_feed PRIVATE disk/feed/cognitive_io.c)
target_link_libraries(aotx_core PUBLIC aotx_cognitive_disk)
target_link_libraries(aotx_restore PRIVATE aotx_ccir_store)
target_link_libraries(aotx_feed PRIVATE aotx_cognitive_disk)
target_link_libraries(aotx_drain PRIVATE aotx_cognitive_disk)
target_compile_options(aotx_cognitive_disk PRIVATE -Wall -Wextra -Werror)
add_executable(aotx_ccir_pack disk/runtime/pack_main.c disk/runtime/pack_assets.c
    disk/runtime/pack_models.c disk/runtime/pack_vision.c disk/runtime/pack_audio.c disk/runtime/pack_state.c
    disk/runtime/pack_appraisal.c)
add_executable(aotx_runtime_publication_test tests/runtime_publication_test.c)
target_link_libraries(aotx_runtime_publication_test PRIVATE aotx_cognitive_disk)
add_test(NAME disk_runtime_publication COMMAND aotx_runtime_publication_test)
add_executable(aotx_runtime_pack_test tests/runtime_pack_test.c disk/runtime/pack_assets.c)
target_link_libraries(aotx_runtime_pack_test PRIVATE aotx_ccir_store)
target_compile_options(aotx_runtime_pack_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME disk_runtime_pack COMMAND aotx_runtime_pack_test)
target_link_libraries(aotx_ccir_pack PRIVATE aotx_cognitive_disk aotx_modelfile aotx_settings)
target_compile_definitions(aotx_ccir_pack PRIVATE AOTX_RUNTIME_ARCH=${AOTX_ARCH}u)
target_compile_options(aotx_ccir_pack PRIVATE -Wall -Wextra -Werror)

add_executable(aotx_ccir_state cuda/cognitive/state_host.cu)
target_link_libraries(aotx_ccir_state PRIVATE aotx_cognitive aotx_cognitive_disk)
set_target_properties(aotx_ccir_state PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_library(aotx_recall_disk STATIC disk/cognitive/recall_io.c)
target_include_directories(aotx_recall_disk PUBLIC disk cuda)
target_link_libraries(aotx_recall_disk PUBLIC aotx_cognitive_disk)
target_compile_options(aotx_recall_disk PRIVATE -Wall -Wextra -Werror)
add_executable(aotx_ccir_recall cuda/cognitive/recall_host.cu)
target_link_libraries(aotx_ccir_recall PRIVATE aotx_cognitive aotx_recall_disk)
set_target_properties(aotx_ccir_recall PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The model file reader is C on the disk side. The boot glue links it to read a model file
# and to check the record of that file. The settings reader is C on the disk side as well.
# The boot glue reads the settings file with it, and the feeder reads the same file.
target_link_libraries(aotx_core PUBLIC aotx_modelfile aotx_settings)

# The interface builds from its own file; it links the window libraries.
add_subdirectory(cuda/ui)

if(AOTX_CTRL)
    add_subdirectory(ctrl)
endif()

# The inert bundle reader is shared by model-file dependency checks and activation.
target_sources(aotx_modelfile PRIVATE disk/policy/file.c disk/policy/write.c disk/policy/history.c
    disk/runtime/policy_read.c disk/runtime/policy_replay.c disk/runtime/policy_update.c)
add_executable(aotx_policy_update disk/policy/update_main.c)
target_link_libraries(aotx_policy_update PRIVATE aotx_modelfile)
target_compile_options(aotx_policy_update PRIVATE -Wall -Wextra -Werror)
add_executable(aotx_policy_pack disk/policy/pack_main.c)
target_link_libraries(aotx_policy_pack PRIVATE aotx_modelfile)
target_compile_options(aotx_policy_pack PRIVATE -Wall -Wextra -Werror)
add_executable(aotx_policy_update_test tests/policy_update_test.c)
target_link_libraries(aotx_policy_update_test PRIVATE aotx_cognitive_disk)
target_link_options(aotx_policy_update_test PRIVATE -Wl,--wrap=linkat)
target_compile_options(aotx_policy_update_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME policy_update COMMAND aotx_policy_update_test)
add_executable(aotx_policy_file_test tests/policy_file_test.c)
target_link_libraries(aotx_policy_file_test PRIVATE aotx_cognitive_disk)
target_compile_options(aotx_policy_file_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME policy_file COMMAND aotx_policy_file_test)
target_sources(aotx_modelfile PRIVATE disk/runtime/appraisal_memory.c disk/runtime/appraisal_assets.c
    disk/runtime/appraisal_replay.c)
target_include_directories(aotx_modelfile PRIVATE ${CMAKE_SOURCE_DIR}/cuda)
add_executable(aotx_appraisal_runtime_disk_test tests/appraisal_runtime_disk_test.c disk/runtime/pack_appraisal.c)
target_link_libraries(aotx_appraisal_runtime_disk_test PRIVATE aotx_cognitive_disk)
target_compile_options(aotx_appraisal_runtime_disk_test PRIVATE -Wall -Wextra -Werror)
add_dependencies(aotx_appraisal_runtime_disk_test aotx_ccir_pack)
add_test(NAME appraisal_runtime_disk COMMAND aotx_appraisal_runtime_disk_test $<TARGET_FILE:aotx_ccir_pack>)
add_executable(aotx_control_disk_test tests/control_disk_test.c)
target_link_libraries(aotx_control_disk_test PRIVATE aotx_cognitive_disk)
target_compile_options(aotx_control_disk_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME control_disk COMMAND aotx_control_disk_test)
