# SPDX-License-Identifier: Apache-2.0
# Set bounded shared table capacities; input is CMake values, output is definitions, invalid values stop configuration.
set(AOTX_SHARED_PARTICIPANTS "1024" CACHE STRING "The persistent participant capacity.")
set(AOTX_SHARED_SPACES "1024" CACHE STRING "The persistent space capacity.")
set(AOTX_SHARED_CONVERSATIONS "4096" CACHE STRING "The persistent conversation capacity.")
set(AOTX_SHARED_MEMBERS "4096" CACHE STRING "The explicit member capacity.")
set(AOTX_SHARED_RECEIPTS "1024" CACHE STRING "The persistent operation receipt capacity.")
set(AOTX_SHARED_COMMAND_BYTES "8192" CACHE STRING "The canonical operation byte capacity.")
set(AOTX_SHARED_RESULT_BYTES "65536" CACHE STRING "The exact output byte capacity per operation.")
foreach(key AOTX_SHARED_PARTICIPANTS AOTX_SHARED_SPACES AOTX_SHARED_CONVERSATIONS AOTX_SHARED_MEMBERS
    AOTX_SHARED_RECEIPTS AOTX_SHARED_COMMAND_BYTES AOTX_SHARED_RESULT_BYTES)
    string(LENGTH "${${key}}" digits)
    if(NOT "${${key}}" MATCHES "^[1-9][0-9]*$" OR digits GREATER 10 OR ${key} GREATER 2147483647)
        message(FATAL_ERROR "${key} exceeds the shared counter range")
    endif()
    target_compile_definitions(aotx_memory_config INTERFACE ${key}=${${key}}u)
endforeach()
if(AOTX_SHARED_COMMAND_BYTES LESS 2560 OR AOTX_SHARED_COMMAND_BYTES GREATER 32768 OR
    AOTX_SHARED_RESULT_BYTES LESS 1024 OR AOTX_SHARED_RESULT_BYTES GREATER 16777216)
    message(FATAL_ERROR "The shared byte capacities exceed the supported range")
endif()
