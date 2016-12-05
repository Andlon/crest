cmake_minimum_required(VERSION 3.2)
project(crest)

include_directories(include/)

set(UNIT_TEST_SOURCES
        tests/unit/unit.cpp
        tests/unit/geometry/indexed_mesh.cpp
        tests/unit/util/vertex_matchers.cpp
        tests/unit/quadrature/triquad.cpp
        tests/unit/basis/lagrange_basis2d.cpp
        tests/unit/basis/quasi_interpolation.cpp
        tests/unit/basis/homogenized_basis.cpp
        tests/unit/util/eigen_extensions.cpp
        tests/unit/geometry/refinement.cpp
        tests/unit/quadrature/simpsons.cpp
        tests/unit/geometry/mesh_algorithms.cpp)

set(INTEGRATION_TEST_SOURCES
        tests/integration/integration.cpp
        tests/integration/convergence.cpp
        tests/integration/io.cpp)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(GCC_CXX_FLAGS "-Wall -Wextra -Werror -march=native")

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

# This should normally only be used by Travis. Lets us use Eigen by simply copying files into external/include/Eigen
include_directories(SYSTEM external/include)

# Add rapidcheck for writing property-based tests
add_subdirectory("external/rapidcheck")

# Add HDF5
find_package(HDF5 COMPONENTS CXX)

# Boost
find_package(Boost 1.58 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Unit tests
add_executable(unit_tests ${UNIT_TEST_SOURCES})
target_link_libraries(unit_tests gmock gmock_main gtest gtest_main rapidcheck)
target_include_directories(unit_tests PRIVATE tests/ SYSTEM external/rapidcheck/extras/gtest/include)
set_property(TARGET unit_tests PROPERTY CXX_STANDARD 14)
add_custom_target(unit COMMAND unit_tests)

# Integration tests
add_executable(integration_tests ${INTEGRATION_TEST_SOURCES})
target_link_libraries(integration_tests gmock gmock_main gtest gtest_main rapidcheck ${HDF5_LIBRARIES})
target_include_directories(integration_tests PRIVATE tests/)
target_include_directories(integration_tests SYSTEM PRIVATE ${HDF5_INCLUDE_DIRS} external/rapidcheck/extras/gtest/include)
set_property(TARGET integration_tests PROPERTY CXX_STANDARD 14)
add_custom_target(integration COMMAND integration_tests)

# Run all tests using the 'check' target
add_custom_target(check COMMAND unit_tests COMMAND integration_tests)

# Pycrest
set(PYCREST_SOURCES pycrest/pycrest.cpp)
add_library(pycrest SHARED ${PYCREST_SOURCES})
set_property(TARGET pycrest PROPERTY CXX_STANDARD 14)

# Simple executables which can be used for profiling with valgrind
add_executable(bench_bisect_to_tolerance bench/bisect_to_tolerance.cpp)
set_property(TARGET bench_bisect_to_tolerance PROPERTY CXX_STANDARD 14)

# Run a set of predefined experiments. Used for testing the library
add_executable(experiments tests/experiment_runner.cpp)
target_include_directories(experiments PRIVATE tests/)
target_include_directories(experiments SYSTEM PRIVATE external/json/src ${HDF5_INCLUDE_DIRS})
target_link_libraries(experiments ${HDF5_LIBRARIES})
set_property(TARGET experiments PROPERTY CXX_STANDARD 14)
