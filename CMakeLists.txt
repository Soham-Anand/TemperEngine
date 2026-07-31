cmake_minimum_required(VERSION 3.14)
project(TemperEngine VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(TEMPER_ROOT ${CMAKE_SOURCE_DIR})
set(TEMPER_INCLUDE_DIR ${TEMPER_ROOT}/include)
set(TEMPER_SRC_DIR ${TEMPER_ROOT}/src)

option(TEMPER_BUILD_TESTS "Build unit tests" ON)
option(TEMPER_BUILD_EXAMPLES "Build examples" ON)
option(TEMPER_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(TEMPER_DEBUG "Enable debug mode" $<CONFIG:Debug>)

add_library(temper STATIC
    ${TEMPER_SRC_DIR}/temper.c

    ${TEMPER_SRC_DIR}/core/memory.c
    ${TEMPER_SRC_DIR}/core/threading.c
    ${TEMPER_SRC_DIR}/core/logger.c
    ${TEMPER_SRC_DIR}/core/profiler.c
    ${TEMPER_SRC_DIR}/core/platform.c
    ${TEMPER_SRC_DIR}/core/device.c
    ${TEMPER_SRC_DIR}/core/resource.c
    ${TEMPER_SRC_DIR}/core/runtime.c

    ${TEMPER_SRC_DIR}/compute/kernel.c

    ${TEMPER_SRC_DIR}/math/tensor.c
    ${TEMPER_SRC_DIR}/math/shape.c

    ${TEMPER_SRC_DIR}/graph/node.c
    ${TEMPER_SRC_DIR}/graph/tape.c
    ${TEMPER_SRC_DIR}/graph/autodiff.c

    ${TEMPER_SRC_DIR}/nn/layers.c
    ${TEMPER_SRC_DIR}/nn/activations.c
    ${TEMPER_SRC_DIR}/nn/losses.c

    ${TEMPER_SRC_DIR}/training/trainer.c
    ${TEMPER_SRC_DIR}/training/optimizer.c
    ${TEMPER_SRC_DIR}/training/checkpoint.c
    ${TEMPER_SRC_DIR}/training/scheduler.c

    ${TEMPER_SRC_DIR}/memory/scheduler.c
    ${TEMPER_SRC_DIR}/memory/compression.c
)

target_include_directories(temper PUBLIC ${TEMPER_INCLUDE_DIR})

if(MSVC)
    target_compile_options(temper PRIVATE
        /W4
        $<$<CONFIG:Debug>:/Zi /Od -DTEMPER_DEBUG>
        $<$<CONFIG:Release>:/O2 -DNDEBUG>
    )
else()
    target_compile_options(temper PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<CONFIG:Debug>:-g -O0 -DTEMPER_DEBUG>
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
    )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(temper PRIVATE pthread m)
    target_sources(temper PRIVATE ${TEMPER_SRC_DIR}/metal/metal_stub.c)
elseif(APPLE)
    enable_language(OBJC)
    target_link_libraries(temper PRIVATE m)
    find_library(FOUNDATION Foundation)
    if(FOUNDATION)
        target_link_libraries(temper PRIVATE ${FOUNDATION})
    endif()
    find_library(METAL_FRAMEWORK Metal)
    if(METAL_FRAMEWORK)
        target_link_libraries(temper PRIVATE ${METAL_FRAMEWORK})
    endif()

    # Metal kernel build pipeline: kernels.metal -> AIR -> metallib -> embedded C.
    add_executable(bin2c tools/bin2c.c)
    set_target_properties(bin2c PROPERTIES EXCLUDE_FROM_ALL TRUE)

    set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/metal/kernels.air)
    set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/metal/kernels.metallib)
    set(METAL_EMBED_C ${CMAKE_CURRENT_BINARY_DIR}/metal/kernels_embedded.c)

    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/metal
        COMMAND xcrun -sdk macosx metal -std=macos-metal2.4 -c
                ${TEMPER_SRC_DIR}/metal/kernels.metal -o ${METAL_AIR}
        DEPENDS ${TEMPER_SRC_DIR}/metal/kernels.metal
        COMMENT "Compiling Metal kernels (AIR)"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking Metal kernels (metallib)"
        VERBATIM
    )
    add_custom_command(
        OUTPUT ${METAL_EMBED_C}
        COMMAND $<TARGET_FILE:bin2c> ${METAL_LIB} ${METAL_EMBED_C} temper_metal_library
        DEPENDS ${METAL_LIB} bin2c
        COMMENT "Embedding Metal library into the static library"
        VERBATIM
    )
    set_source_files_properties(${METAL_EMBED_C} PROPERTIES GENERATED TRUE)
    target_sources(temper PRIVATE ${METAL_EMBED_C})

    target_sources(temper PRIVATE ${TEMPER_SRC_DIR}/metal/metal.m)
    set_source_files_properties(${TEMPER_SRC_DIR}/metal/metal.m
        PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_compile_definitions(temper PUBLIC TEMPER_HAS_METAL)
elseif(WIN32)
    target_link_libraries(temper PRIVATE winmm)
    target_sources(temper PRIVATE ${TEMPER_SRC_DIR}/metal/metal_stub.c)
endif()

if(TEMPER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TEMPER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(TEMPER_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

include(GNUInstallDirs)

install(TARGETS temper
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/temper DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
