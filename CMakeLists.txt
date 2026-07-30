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
option(TEMPER_DEBUG "Enable debug mode" $<CONFIG:Debug>)

add_library(temper STATIC
    ${TEMPER_SRC_DIR}/temper.c

    ${TEMPER_SRC_DIR}/core/memory.c
    ${TEMPER_SRC_DIR}/core/threading.c
    ${TEMPER_SRC_DIR}/core/logger.c
    ${TEMPER_SRC_DIR}/core/profiler.c
    ${TEMPER_SRC_DIR}/core/platform.c

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
elseif(APPLE)
    target_link_libraries(temper PRIVATE m)
    find_library(FOUNDATION Foundation)
    if(FOUNDATION)
        target_link_libraries(temper PRIVATE ${FOUNDATION})
    endif()
elseif(WIN32)
    target_link_libraries(temper PRIVATE winmm)
endif()

if(TEMPER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TEMPER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

include(GNUInstallDirs)

install(TARGETS temper
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/temper DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
