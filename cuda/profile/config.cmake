# The maximum keeps a 32-bit token index valid at all 256 supported slots.
# The sequence capacity is independent of the card profile and physical page pool.
set(AOTX_SEQUENCE_TOKENS "0" CACHE STRING "The prompt and reply token capacity; zero uses the profile.")
string(LENGTH "${AOTX_SEQUENCE_TOKENS}" digits)
if(NOT AOTX_SEQUENCE_TOKENS MATCHES "^(0|[1-9][0-9]*)$" OR digits GREATER 8 OR
   AOTX_SEQUENCE_TOKENS GREATER 16777215 OR AOTX_SEQUENCE_TOKENS EQUAL 1)
    message(FATAL_ERROR "AOTX_SEQUENCE_TOKENS must be zero or an integer from 2 through 16777215")
endif()
target_compile_definitions(aotx_memory_config INTERFACE AOTX_SEQUENCE_TOKENS=${AOTX_SEQUENCE_TOKENS}u)
