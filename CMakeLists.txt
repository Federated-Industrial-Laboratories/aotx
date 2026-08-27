# The build for aotx. Device code targets compute capability 8.6.
cmake_minimum_required(VERSION 3.28)
project(aotx LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
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
    cuda/sched/step.cu
    cuda/sched/pump_host.cu
    cuda/boot/clock_host.cu
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
    cuda/cli/editor.cu
    cuda/cli/parse.cu)
target_include_directories(aotx_core PUBLIC cuda)
target_link_libraries(aotx_core PUBLIC ${AOTX_LIBCUDA})
target_compile_definitions(aotx_core PUBLIC AOTX_PTX_DIR="${CMAKE_SOURCE_DIR}/ptx")

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
# and to check the record of that file.
target_link_libraries(aotx_core PUBLIC aotx_modelfile)

# The interface builds from its own file; it links the window libraries.
add_subdirectory(cuda/ui)
