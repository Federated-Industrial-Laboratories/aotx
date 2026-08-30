# The model directory is an input of each check that reads model files.
set(AOTX_MODELS_DIR ${CMAKE_SOURCE_DIR}/models CACHE STRING "Directory that holds the model files.")

add_test(NAME size_gate_boundary COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/size_gate_test.py
         ${CMAKE_SOURCE_DIR}/tools/size-gate.py)

# The test programs. A test is a host program that launches kernels. A device test target
# carries a device mark, because the disk side has a test of the same subject.
foreach(name rng mem seam bus bulk kvcache sched model sample settings load)
    add_executable(aotx_${name}_device_test ${name}_test.cu)
    target_link_libraries(aotx_${name}_device_test PRIVATE aotx_core Threads::Threads)
    set_target_properties(aotx_${name}_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    if(name STREQUAL "load")
        add_test(NAME ${name} COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh
                 ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_${name}_device_test>
                 ${AOTX_MODELS_DIR})
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
    else()
        add_test(NAME ${name} COMMAND aotx_${name}_device_test)
    endif()
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
add_executable(aotx_text_device_test text_test.cu)
target_link_libraries(aotx_text_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_text_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME text COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_text_device_test>
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer ${AOTX_MODELS_DIR})
set_tests_properties(text PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)

# The matrix check reads the model files for the real weight rows. A file which is not
# there gives a skip that the check reports as a figure.
add_executable(aotx_matrix_device_test matrix_test.cu)
target_link_libraries(aotx_matrix_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_matrix_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME matrix COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_matrix_device_test> ${AOTX_MODELS_DIR})
set_tests_properties(matrix PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)

# The model gate reads the reference lists and the model files. The model files are large
# and are not in the repository, so the gate reports a skip for a directory which is empty.
add_executable(aotx_model_gate_device_test model_gate_test.cu)
target_link_libraries(aotx_model_gate_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_model_gate_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME model_gate COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh
         ${AOTX_MODELS_DIR} $<TARGET_FILE:aotx_model_gate_device_test>
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model ${AOTX_MODELS_DIR})
set_tests_properties(model_gate PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)

# The decode check runs the tick graph with a language model. The model files are large and
# are not in the repository, so the check reports a skip for a directory which is empty.
add_executable(aotx_decode_device_test decode_test.cu)
target_link_libraries(aotx_decode_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_decode_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME decode COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_decode_device_test> ${AOTX_MODELS_DIR}
         ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer)
set_tests_properties(decode PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)

# The tool check runs the parser with no model and the two memory tools with the embedding
# model. The model files are large and are not in the repository, so the check reports a
# skip for a directory which is empty.
add_executable(aotx_tool_device_test tool_test.cu)
target_link_libraries(aotx_tool_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_tool_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME tool COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_tool_device_test> ${AOTX_MODELS_DIR})
set_tests_properties(tool PROPERTIES TIMEOUT 900 SKIP_RETURN_CODE 77)

# The agent check runs the turn loop on the real model. It reads the model directory and
# the module directories, because the roles of a run come in as modules.
add_executable(aotx_agent_device_test agent_test.cu)
target_link_libraries(aotx_agent_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_agent_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME agent COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
         $<TARGET_FILE:aotx_agent_device_test> ${AOTX_MODELS_DIR}
         ${CMAKE_SOURCE_DIR}/modules/roles)
set_tests_properties(agent PROPERTIES TIMEOUT 2400 SKIP_RETURN_CODE 77)

# The conversation check drives line parts through the apply and checks the tier state.
add_executable(aotx_conversation_device_test conversation_test.cu)
target_link_libraries(aotx_conversation_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_conversation_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME conversation COMMAND aotx_conversation_device_test)
set_tests_properties(conversation PROPERTIES TIMEOUT 600)

# The catalog check drives the import path over the inbound ring and the apply node. It
# needs no model file, so it runs on every build.
add_executable(aotx_catalog_device_test catalog_test.cu)
target_link_libraries(aotx_catalog_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_catalog_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME catalog COMMAND aotx_catalog_device_test ${CMAKE_SOURCE_DIR}/modules)
set_tests_properties(catalog PROPERTIES TIMEOUT 900)

# The module check drives the device tool modules. It takes the import, the node of the
# tick graph and the capture that follows an import. It takes the batch at 1 row and at the
# row count, and the refusal of a module file that changed on disk. It reads the module
# directory the setup built, so it needs no model file.
add_executable(aotx_module_device_test module_test.cu)
target_link_libraries(aotx_module_device_test PRIVATE aotx_core Threads::Threads)
set_target_properties(aotx_module_device_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# The example modules and the fixtures of the check program go into the build directory
# and are built there. A check therefore leaves the source tree as it was.
add_test(NAME module_setup
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_setup.sh ${CMAKE_SOURCE_DIR}
                 ${CMAKE_BINARY_DIR} ${AOTX_ARCH})
set_tests_properties(module_setup PROPERTIES FIXTURES_SETUP module_files TIMEOUT 600)

add_test(NAME module COMMAND aotx_module_device_test ${CMAKE_BINARY_DIR}/modules/word_count
         ${CMAKE_BINARY_DIR})
set_tests_properties(module PROPERTIES FIXTURES_REQUIRED module_files TIMEOUT 900)

# The check program over the two example modules. With no row count the check program runs
# the module at one row and at the row count of the profile.
add_test(NAME example_word_count
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_check.sh
                 $<TARGET_FILE:aotx_module_check>
                 ${CMAKE_BINARY_DIR}/modules/word_count pass)
set_tests_properties(example_word_count PROPERTIES FIXTURES_REQUIRED module_files
                     TIMEOUT 300)

add_test(NAME example_echo_upper
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_check.sh
                 $<TARGET_FILE:aotx_module_check>
                 ${CMAKE_BINARY_DIR}/modules/echo_upper pass)
set_tests_properties(example_echo_upper PROPERTIES FIXTURES_REQUIRED module_files
                     TIMEOUT 300)

# Each refusal of the check program is proven by a fixture module that earns it.
foreach(case local_memory wrong_entry over_bound untaken_row unknown_key hang_tool
             bad_digest)
    add_test(NAME module_refuse_${case}
             COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_check.sh
                     $<TARGET_FILE:aotx_module_check>
                     ${CMAKE_BINARY_DIR}/modules/${case} fail)
    set_tests_properties(module_refuse_${case} PROPERTIES FIXTURES_REQUIRED module_files
                         TIMEOUT 300)
endforeach()

# The program of hang_tool never ends by itself. The check program must end it at the
# timeout of its manifest, which is one second. A short allowance here therefore fails a
# check program that waits for such a program instead of ending it.
set_tests_properties(module_refuse_hang_tool PROPERTIES TIMEOUT 60)

# The seam gate of the build script is proven able to fail: a module that holds a host call
# does not become a module file.
add_test(NAME module_build_host_call
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/module_build.sh ${CMAKE_SOURCE_DIR}
                 ${CMAKE_BINARY_DIR}/modules/host_call fail ${AOTX_ARCH})
set_tests_properties(module_build_host_call PROPERTIES FIXTURES_REQUIRED module_files
                     TIMEOUT 300)

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
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/model_test.sh ${AOTX_MODELS_DIR}
                 ${CMAKE_CURRENT_SOURCE_DIR}/replay_test.sh
                 ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/replay-journal ${AOTX_MODELS_DIR})
set_tests_properties(replay PROPERTIES TIMEOUT 2400 FIXTURES_REQUIRED module_files
                     SKIP_RETURN_CODE 77)

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

# The disk model-store targets are declared by disk/CMakeLists.txt after this directory.
add_test(NAME disk_catalog COMMAND $<TARGET_FILE:aotx_catalog_test>
         ${CMAKE_SOURCE_DIR}/share/models/catalog.jsonl)
add_test(NAME disk_store COMMAND $<TARGET_FILE:aotx_store_test>)
add_test(NAME disk_fetch COMMAND $<TARGET_FILE:aotx_fetch_test>)
add_test(NAME disk_models COMMAND $<TARGET_FILE:aotx_models_test> $<TARGET_FILE:aotx_models>)
set_tests_properties(disk_fetch PROPERTIES TIMEOUT 120)
add_test(NAME disk_feed_models COMMAND $<TARGET_FILE:aotx_feed_models_test>
         $<TARGET_FILE:aotx_feed>)
set_tests_properties(disk_feed_models PROPERTIES TIMEOUT 30)
