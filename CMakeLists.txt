# @file BDBtTiming/CMakeList.txt  -- First file to read for CMake

cmake_minimum_required(VERSION 3.8...3.15)

project (BDBTtiming LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Finding the pz package
find_package(PZ REQUIRED)
# Finding the blaze package
find_package( blaze REQUIRED)
if( blaze_FOUND )
    add_library( blaze_target INTERFACE )
    target_link_libraries( blaze_target INTERFACE blaze::blaze )
endif()

add_subdirectory(MassElementMatrix)
