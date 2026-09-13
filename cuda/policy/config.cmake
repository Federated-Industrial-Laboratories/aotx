# SPDX-License-Identifier: Apache-2.0
# Set independent policy state and image byte capacities for the selected build.
set(AOTX_POLICY_STATE_BYTES "65536" CACHE STRING "The maximum persistent bytes of one creator policy.")
set(AOTX_POLICY_IMAGE_BYTES "16777216" CACHE STRING "The maximum bytes of one creator native image.")
foreach(value AOTX_POLICY_STATE_BYTES AOTX_POLICY_IMAGE_BYTES)
    string(LENGTH "${${value}}" digits)
    if(NOT "${${value}}" MATCHES "^[1-9][0-9]*$" OR digits GREATER 10 OR
       ${value} GREATER 2147483391 OR ${value} LESS 16)
        message(FATAL_ERROR "${value} is outside the policy byte-count range")
    endif()
    target_compile_definitions(aotx_memory_config INTERFACE ${value}=${${value}}u)
endforeach()
