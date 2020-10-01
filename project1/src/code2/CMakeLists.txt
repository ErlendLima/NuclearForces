cmake_minimum_required(VERSION 3.10)

project(rmatrix VERSION 0.0)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

#link_directories(SFML_LIBRARY_DIR)
#include_directories(SFML_INCLUDE_DIR)

#find_package(SFML 2.5.1 COMPONENTS system window graphics)

add_executable(main code.cpp)
target_link_libraries(main ${ARMADILLO_LIBRARIES})
add_compile_options(-Wall -Wextra -pedantic -Werror)
