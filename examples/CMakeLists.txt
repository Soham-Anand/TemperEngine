add_executable(hello hello.c)
target_link_libraries(hello temper)

add_executable(metal metal.c)
target_link_libraries(metal temper)
