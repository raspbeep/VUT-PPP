##
# @file    CMakeLists.txt
# @authors Filip Vaverka <ivaverka@fit.vutbr.cz>
#          Jiri Jaros <jarosjir@fit.vutbr.cz>
#          David Bayer <ibayer@fit.vutbr.cz>
#
# @brief   Course: PPP 2023/2024 - Project 1
#          CMake build system project specifications.
#
# @date    2024-02-22
##

cmake_minimum_required(VERSION 3.22)

project(ppp_proj01 LANGUAGES C CXX)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Set default options
set(ALLOW_OPENMP ON CACHE BOOL "Allow OpenMP for multi-threading")

# Enable unity build
set(CMAKE_UNITY_BUILD ON)

# Find MPI package
set(MPI_CXX_SKIP_MPICXX ON) # Do not use mpicxx wrapper
find_package(MPI REQUIRED COMPONENTS CXX)

# Find OpenMP package
if((NOT MSVC) AND ${ALLOW_OPENMP})
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Find HDF5 package
set(HDF5_PREFER_PARALLEL ON) # Prefer parallel version of HDF5
find_package(HDF5 REQUIRED COMPONENTS C HL)

# Set compiler flags
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
   (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang") OR
   ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND ((CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"))))
  set(CMAKE_CXX_FLAGS                  "-Wall -Wextra")
  set(CMAKE_CXX_FLAGS_DEBUG            "-O0 -g3")
  set(CMAKE_CXX_FLAGS_RELEASE          "-Ofast -march=native -mtune=native -fassociative-math -flto=auto")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO   "${CMAKE_CXX_FLAGS_RELEASE} -g1")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  if(WIN32)
    set(CMAKE_CXX_FLAGS                "/W4")
    set(CMAKE_CXX_FLAGS_DEBUG          "/Od /debug:full")
    set(CMAKE_CXX_FLAGS_RELEASE        "/fast /MT /QxHost") # -flto=full is well formed here, see docs!
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} /debug:minimal")
  else()
    set(CMAKE_CXX_FLAGS                "-Wall -Xlinker -lstdc++")
    set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -debug full")
    set(CMAKE_CXX_FLAGS_RELEASE        "-ipo -O3 -no-prec-div -fp-model=fast=2 -xHost")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -debug minimal -no-ipo")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS                "/W4 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG          "/Od /debug:full")
    set(CMAKE_CXX_FLAGS_RELEASE        "-O3 /QxHost /fp:fast") # -flto=full is well formed here, see docs!
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} /debug:minimal")
  elseif(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS                "-Wall -Wno-unused-command-line-argument -Xlinker -lstdc++")
    set(CMAKE_CXX_FLAGS_DEBUG          "-O0 -debug full")
    set(CMAKE_CXX_FLAGS_RELEASE        "-flto=full -fast -xHost")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -debug minimal")
  else()
    message(WARNING "Unsupported IntelLLVM compiler frontend variant: ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
  endif()
elseif((CMAKE_CXX_COMPILER_ID STREQUAL "MSVC") OR
       ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")))
  set(CMAKE_CXX_FLAGS                  "/W4 /Zc:__cplusplus /EHsc")
  set(CMAKE_CXX_FLAGS_DEBUG            "/Od /Z7")
  set(CMAKE_CXX_FLAGS_RELEASE          "/Ox /fp:fast /GL")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO   "${CMAKE_CXX_FLAGS_RELEASE} /Z7")

  set(CMAKE_SHARED_LINKER_FLAGS        "/LTCG")
else()
  message(WARNING "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()

# Add cxxopts
add_subdirectory(3rdparty/cxxopts-3.2.0 EXCLUDE_FROM_ALL)

# Add fmtlib
set(FMT_INSTALL        OFF)
set(FMT_SYSTEM_HEADERS ON)
add_subdirectory(3rdparty/fmt-10.2.1 EXCLUDE_FROM_ALL)

# Solver executable
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/sources/main.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/sources/HeatSolverBase.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/sources/MaterialProperties.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/sources/ParallelHeatSolver.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/sources/SequentialHeatSolver.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/sources/SimulationProperties.cpp
                               ${CMAKE_CURRENT_SOURCE_DIR}/sources/utils.cpp)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/stb)
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX
                                              HDF5::HDF5
                                              cxxopts::cxxopts
                                              fmt::fmt-header-only)
if((NOT MSVC) AND ${ALLOW_OPENMP})
  target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
endif()

# Data generator
add_executable(data_generator  ${CMAKE_CURRENT_SOURCE_DIR}/sources/data_generator.cpp)
target_link_libraries(data_generator PRIVATE HDF5::HDF5 hdf5::hdf5_hl
                                             cxxopts::cxxopts
                                             fmt::fmt-header-only)
if((NOT MSVC) AND ${ALLOW_OPENMP})
  target_link_libraries(data_generator PRIVATE OpenMP::OpenMP_CXX)
endif()

# A simple test target
add_custom_target(test DEPENDS ${PROJECT_NAME} data_generator)
add_custom_command(TARGET test POST_BUILD
  COMMAND ./data_generator -n 256 -o test_input.h5
  COMMAND ./${PROJECT_NAME} -n 100 -m 0 -w 10 -i test_input.h5 -o result.h5)

# Create zip archive with student's sources and documentation if LOGIN was defined
if(NOT DEFINED LOGIN)
  message(WARNING "LOGIN variable is not defined, you will not be able to create final archive (set by '-DLOGIN=...').")
else()
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${LOGIN})
  add_custom_target(pack 
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/sources/ParallelHeatSolver.hpp ${CMAKE_CURRENT_BINARY_DIR}/${LOGIN}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/sources/ParallelHeatSolver.cpp ${CMAKE_CURRENT_BINARY_DIR}/${LOGIN}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${LOGIN}.pdf ${CMAKE_CURRENT_BINARY_DIR}/${LOGIN}
    COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/${LOGIN}.zip" --format=zip
                                         "ParallelHeatSolver.hpp"
                                         "ParallelHeatSolver.cpp"
                                         "${LOGIN}.pdf"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${LOGIN})
endif()

# Create zip archive with sources for assignment if the assignment sources exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/sources/assignment)
  set(ASSIGNMENT_SRC_FILES_LIST ${CMAKE_CURRENT_SOURCE_DIR}/sources/AlignedAllocator.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/data_generator.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/Hdf5Handle.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/HeatSolverBase.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/HeatSolverBase.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/main.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/MaterialProperties.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/MaterialProperties.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/assignment/ParallelHeatSolver.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/assignment/ParallelHeatSolver.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/SequentialHeatSolver.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/SequentialHeatSolver.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/SimulationProperties.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/SimulationProperties.hpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/utils.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/sources/utils.hpp)

  add_custom_target(assignment COMMAND
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/assignment
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty ${CMAKE_CURRENT_BINARY_DIR}/assignment/3rdparty
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/scripts ${CMAKE_CURRENT_BINARY_DIR}/assignment/scripts
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/doc/proj01.pdf ${CMAKE_CURRENT_BINARY_DIR}/assignment
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt ${CMAKE_CURRENT_BINARY_DIR}/assignment
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/assignment/sources
    COMMAND ${CMAKE_COMMAND} -E copy ${ASSIGNMENT_SRC_FILES_LIST} ${CMAKE_CURRENT_BINARY_DIR}/assignment/sources
    COMMAND ${CMAKE_COMMAND} -E tar "cf" "${CMAKE_CURRENT_BINARY_DIR}/ppp_proj01.zip" --format=zip
                                         "${CMAKE_CURRENT_BINARY_DIR}/assignment"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
