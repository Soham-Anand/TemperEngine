set(TEST_SOURCES
    test_core.c
    test_math.c
    test_nn.c
    test_tensor_mobility.c
    test_memory_scheduler.c
    test_runtime.c
)

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} temper)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
