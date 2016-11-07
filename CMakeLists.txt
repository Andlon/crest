cmake_minimum_required(VERSION 3.2)
project(crest)

include_directories(include/)

set(UNIT_TEST_SOURCES
        tests/unit/unit.cpp
        tests/unit/geometry/indexed_mesh.cpp
        tests/unit/util/vertex_matchers.cpp
        tests/unit/quadrature/triquad.cpp
        tests/unit/basis/linear_lagrangian_basis.cpp
        tests/unit/basis/quasi_interpolation.cpp
        tests/unit/basis/homogenized_basis.cpp
        tests/unit/util/eigen_extensions.cpp
        tests/unit/geometry/refinement.cpp
        tests/unit/quadrature/simpsons.cpp
        tests/unit/geometry/mesh_algorithms.cpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(GCC_CXX_FLAGS "-Wall -Wextra -Werror")

    set(GCC_SANITIZE_FLAGS "-fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover")
    set(GCC_STD_DEBUG "-D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC")
    set(GCC_EXTRA_DEBUG "-ggdb3")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${GCC_SANITIZE_FLAGS} ${GCC_STD_DEBUG} ${GCC_EXTRA_DEBUG}")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
endif()

# Add googletest as part of the build
add_subdirectory(external/googletest)

# Make GTEST a SYSTEM include directory, to prevent warning propagation etc.
include_directories(SYSTEM GTEST_ROOT/include)

# Add rapidcheck for writing property-based tests
add_subdirectory("external/rapidcheck")

# Unit tests
add_executable(unit_tests ${UNIT_TEST_SOURCES})
target_link_libraries(unit_tests gmock gmock_main gtest gtest_main rapidcheck)
target_include_directories(unit_tests PRIVATE tests/ SYSTEM external/rapidcheck/extras/gtest/include)
set_property(TARGET unit_tests PROPERTY CXX_STANDARD 14)
add_custom_target(unit COMMAND unit_tests)

# Run all tests using the 'check' target
add_custom_target(check COMMAND unit_tests)

# Pycrest
set(PYCREST_SOURCES pycrest/pycrest.cpp)
add_library(pycrest SHARED ${PYCREST_SOURCES})
set_property(TARGET pycrest PROPERTY CXX_STANDARD 14)

# Simple executables which can be used for profiling with valgrind
add_executable(bench_bisect_to_tolerance bench/bisect_to_tolerance.cpp)
set_property(TARGET bench_bisect_to_tolerance PROPERTY CXX_STANDARD 14)
