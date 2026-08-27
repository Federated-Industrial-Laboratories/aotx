# The test programs. A test is a host program that launches kernels.
foreach(name rng mem seam)
    add_executable(aotx_${name}_test ${name}_test.cu)
    target_link_libraries(aotx_${name}_test PRIVATE aotx_core Threads::Threads)
    set_target_properties(aotx_${name}_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME ${name} COMMAND aotx_${name}_test)
endforeach()

# The replay check runs the system, kills it, restores it, and compares the state hash.
add_test(NAME replay
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/replay_test.sh
                 ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/replay-journal)
set_tests_properties(replay PROPERTIES TIMEOUT 300)
