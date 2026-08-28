# The test programs. A test is a host program that launches kernels. A device test target
# carries a device mark, because the disk side has a test of the same subject.
foreach(name rng mem seam bus bulk kvcache sched model sample settings)
    add_executable(aotx_${name}_device_test ${name}_test.cu)
    target_link_libraries(aotx_${name}_device_test PRIVATE aotx_core Threads::Threads)
    set_target_properties(aotx_${name}_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    add_test(NAME ${name} COMMAND aotx_${name}_device_test)
endforeach()

# The profile check runs the rule that states the memory a profile needs. It opens no
# context and needs no card, so it runs on every build of every profile.
add_executable(aotx_profile_test profile_test.cu)
target_link_libraries(aotx_profile_test PRIVATE aotx_core)
set_target_properties(aotx_profile_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME profile COMMAND aotx_profile_test)

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

# The matrix check reads the model files for the real weight rows. A file which is not
# there gives a skip that the check reports as a figure.
add_executable(aotx_matrix_device_test matrix_test.cu)
target_link_libraries(aotx_matrix_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_matrix_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME matrix COMMAND aotx_matrix_device_test ${AOTX_MODELS_DIR})
set_tests_properties(matrix PROPERTIES TIMEOUT 600)

# The model gate reads the reference lists and the model files. The model files are large
# and are not in the repository, so the gate reports a skip for a directory which is empty.
add_executable(aotx_model_gate_device_test model_gate_test.cu)
target_link_libraries(aotx_model_gate_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_model_gate_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME model_gate COMMAND aotx_model_gate_device_test
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model ${AOTX_MODELS_DIR})
set_tests_properties(model_gate PROPERTIES TIMEOUT 1800)

# The decode check runs the tick graph with a language model. The model files are large and
# are not in the repository, so the check reports a skip for a directory which is empty.
add_executable(aotx_decode_device_test decode_test.cu)
target_link_libraries(aotx_decode_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_decode_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME decode COMMAND aotx_decode_device_test ${AOTX_MODELS_DIR}
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer)
set_tests_properties(decode PROPERTIES TIMEOUT 1800)

# The tool check runs the parser with no model and the two memory tools with the embedding
# model. The model files are large and are not in the repository, so the check reports a
# skip for a directory which is empty.
add_executable(aotx_tool_device_test tool_test.cu)
target_link_libraries(aotx_tool_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_tool_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME tool COMMAND aotx_tool_device_test ${AOTX_MODELS_DIR})
set_tests_properties(tool PROPERTIES TIMEOUT 900)

# The agent check runs the turn loop on the real model. It reads the model directory.
add_executable(aotx_agent_device_test agent_test.cu)
target_link_libraries(aotx_agent_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_agent_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME agent COMMAND aotx_agent_device_test ${AOTX_MODELS_DIR})
set_tests_properties(agent PROPERTIES TIMEOUT 2400)

# The spill gate reads the resource usage the device linker wrote into the built objects.
# It refuses local memory or a stack frame over the allowance in the hot kernels. The
# figures come from cuobjdump, which stands beside the compiler of this build.
get_filename_component(AOTX_CUDA_BIN ${CMAKE_CUDA_COMPILER} DIRECTORY)
add_test(NAME spill
         COMMAND ${CMAKE_SOURCE_DIR}/tools/spill-gate.py
                 --cuobjdump ${AOTX_CUDA_BIN}/cuobjdump ${CMAKE_BINARY_DIR})

# The graph check and the tick check take longer than a unit test, because the tick check
# measures 500 ticks.
set_tests_properties(sched PROPERTIES TIMEOUT 300)

# The replay check runs the system, kills it, restores it, and compares the state hash. The
# second scenario asks the language model for a reply. It reads the model directory and
# loads the weights twice, so the time allowance holds both loads.
add_test(NAME replay
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/replay_test.sh
                 ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/replay-journal ${AOTX_MODELS_DIR})
set_tests_properties(replay PROPERTIES TIMEOUT 2400)

# The sanitizer gate runs compute-sanitizer over the checks that drive the tick graph. A
# run takes minutes, because the sanitizer makes every kernel far slower. The gate is
# registered only when a build asks for it: cmake -DAOTX_SANITIZER_TESTS=ON.
option(AOTX_SANITIZER_TESTS "Register the compute-sanitizer gate" OFF)
if(AOTX_SANITIZER_TESTS)
    foreach(tool memcheck racecheck)
        add_test(NAME sanitizer_${tool}
                 COMMAND ${CMAKE_SOURCE_DIR}/tools/sanitizer-gate.sh ${tool}
                         ${CMAKE_BINARY_DIR} ${AOTX_MODELS_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer)
        set_tests_properties(sanitizer_${tool} PROPERTIES TIMEOUT 7200 RUN_SERIAL TRUE
                             LABELS sanitizer
                             ENVIRONMENT AOTX_SANITIZER_BIN=${AOTX_CUDA_BIN}/compute-sanitizer)
    endforeach()
endif()

# The fault cases write past a guard gap on purpose. The fault kills a context on the display
# device, so they run only when this option is on: cmake -DAOTX_FAULT_TESTS=ON.
option(AOTX_FAULT_TESTS "Run the tests that fault the device on purpose" OFF)
if(AOTX_FAULT_TESTS)
    foreach(name mem kvcache)
        add_test(NAME ${name}_fault COMMAND aotx_${name}_device_test)
        set_tests_properties(${name}_fault PROPERTIES ENVIRONMENT AOTX_FAULT_TESTS=1 RUN_SERIAL TRUE)
    endforeach()
endif()
