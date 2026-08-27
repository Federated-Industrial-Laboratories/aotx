# The test programs. A test is a host program that launches kernels. A device test target
# carries a device mark, because the disk side has a test of the same subject.
foreach(name rng mem seam bus bulk kvcache sched)
    add_executable(aotx_${name}_device_test ${name}_test.cu)
    target_link_libraries(aotx_${name}_device_test PRIVATE aotx_core Threads::Threads)
    set_target_properties(aotx_${name}_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME ${name} COMMAND aotx_${name}_device_test)
endforeach()

# The tokenizer check reads the fixture lists and the tokenizer of a model file. The model
# files are large and are not in the repository, so the check reports a skip for a file
# which is not there. The directory is the one the disk side checks read.
set(AOTX_MODELS_DIR ${CMAKE_SOURCE_DIR}/models CACHE STRING "Directory that holds the model files.")
add_executable(aotx_text_device_test text_test.cu)
target_link_libraries(aotx_text_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_text_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME text COMMAND aotx_text_device_test
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer ${AOTX_MODELS_DIR})
set_tests_properties(text PROPERTIES TIMEOUT 900)

# The graph check and the tick check take longer than a unit test, because the tick check
# measures 500 ticks.
set_tests_properties(sched PROPERTIES TIMEOUT 300)

# The replay check runs the system, kills it, restores it, and compares the state hash.
add_test(NAME replay
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/replay_test.sh
                 ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/replay-journal)
set_tests_properties(replay PROPERTIES TIMEOUT 300)

# The fault cases write past a guard gap on purpose. The fault kills a context on the display
# device, so they run only when this option is on: cmake -DAOTX_FAULT_TESTS=ON.
option(AOTX_FAULT_TESTS "Run the tests that fault the device on purpose" OFF)
if(AOTX_FAULT_TESTS)
    foreach(name mem kvcache)
        add_test(NAME ${name}_fault COMMAND aotx_${name}_device_test)
        set_tests_properties(${name}_fault PROPERTIES ENVIRONMENT AOTX_FAULT_TESTS=1 RUN_SERIAL TRUE)
    endforeach()
endif()
