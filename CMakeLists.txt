# The build for aotx. A profile fixes every figure that sizes a device table, and the
# architecture option names the compute capability the device code targets.
cmake_minimum_required(VERSION 3.28)
project(aotx VERSION 0.1.0 LANGUAGES C CXX CUDA)

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
    cuda/model/sample.cu
    cuda/model/decode.cu
    cuda/model/plan.cu
    cuda/model/commit.cu
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
    cuda/boot/clock_host.cu
    cuda/boot/card_host.cu
    cuda/boot/options_host.cu
    cuda/boot/children_host.cu
    cuda/boot/models_host.cu
    cuda/boot/weights_host.cu
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
    cuda/agent/step.cu
    cuda/agent/agent_host.cu
    cuda/tool/parse.cu
    cuda/tool/host_tools.cu
    cuda/tool/device_tools.cu
    cuda/tool/tool_host.cu
    cuda/embed/search.cu
)
target_include_directories(aotx_core PUBLIC cuda)
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
add_executable(aotx_boot cuda/boot/boot_host.cu)
target_link_libraries(aotx_boot PRIVATE aotx_core)
set_target_properties(aotx_boot PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

enable_testing()
# The clock module check keeps the module path proven at every build.
add_test(NAME clock_module COMMAND aotx_boot --clock-only)

add_subdirectory(tests)

# The disk side builds from its own file.
add_subdirectory(disk)

# The model file reader is C on the disk side. The boot glue links it to read a model file
# and to check the record of that file. The settings reader is C on the disk side as well.
# The boot glue reads the settings file with it, and the feeder reads the same file.
target_link_libraries(aotx_core PUBLIC aotx_modelfile aotx_settings)

# The interface builds from its own file; it links the window libraries.
add_subdirectory(cuda/ui)
