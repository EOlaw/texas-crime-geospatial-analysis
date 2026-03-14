cmake_minimum_required(VERSION 3.18)
project(texas_crime_spatial VERSION 1.0.0 LANGUAGES CXX)

# ── Standards ────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type ───────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Optimisation flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

# ── Find Python & pybind11 ───────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Try to find pybind11 via CMake config (works after `pip install pybind11`)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_RESULT
)

if(PYBIND11_RESULT EQUAL 0)
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  find_package(pybind11 CONFIG REQUIRED)
else()
  message(FATAL_ERROR
    "pybind11 not found. Install it with:\n"
    "  pip install pybind11\n"
    "then re-run cmake.")
endif()

# ── Source files ─────────────────────────────────────────────────────────────
set(KDTREE_SOURCES
    src/cpp/spatial_index/kdtree.cpp
    src/cpp/spatial_index/kdtree.h
)

set(DBSCAN_SOURCES
    src/cpp/clustering/dbscan.cpp
    src/cpp/clustering/dbscan.h
)

set(BINDING_SOURCES
    src/cpp/bindings/spatial_ext.cpp
)

# ── Static library: spatial core ─────────────────────────────────────────────
add_library(spatial_core STATIC
    ${KDTREE_SOURCES}
    ${DBSCAN_SOURCES}
)

target_include_directories(spatial_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src/cpp
)

target_compile_options(spatial_core PRIVATE
    $<$<CONFIG:Release>:-O3>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# ── pybind11 Python extension ─────────────────────────────────────────────────
pybind11_add_module(texas_crime_spatial ${BINDING_SOURCES})

target_include_directories(texas_crime_spatial PRIVATE
    ${CMAKE_SOURCE_DIR}/src/cpp
)

target_link_libraries(texas_crime_spatial PRIVATE spatial_core)

# Output the .so next to the project root so Python can import it directly
set_target_properties(texas_crime_spatial PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# ── Unit tests (C++) ─────────────────────────────────────────────────────────
enable_testing()

add_executable(test_kdtree tests/cpp/test_kdtree.cpp)
target_include_directories(test_kdtree PRIVATE ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(test_kdtree PRIVATE spatial_core)
add_test(NAME KDTreeTests COMMAND test_kdtree)

add_executable(test_dbscan tests/cpp/test_dbscan.cpp)
target_include_directories(test_dbscan PRIVATE ${CMAKE_SOURCE_DIR}/src/cpp)
target_link_libraries(test_dbscan PRIVATE spatial_core)
add_test(NAME DBSCANTests COMMAND test_dbscan)

# ── Install ──────────────────────────────────────────────────────────────────
install(TARGETS texas_crime_spatial
        LIBRARY DESTINATION ${Python3_SITEARCH})

# ── Summary ──────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "  texas_crime_spatial  v${PROJECT_VERSION}")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Python      : ${Python3_VERSION}")
message(STATUS "  pybind11    : ${pybind11_VERSION}")
message(STATUS "  Output dir  : ${CMAKE_SOURCE_DIR}")
message(STATUS "═══════════════════════════════════════════")
message(STATUS "")
