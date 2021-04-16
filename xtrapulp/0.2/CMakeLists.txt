cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(graph_to_binary)

find_package(Torch REQUIRED)

add_executable(graph_to_binary graph_to_binary.cpp)
target_link_libraries(graph_to_binary "${TORCH_LIBRARIES}")
set_property(TARGET graph_to_binary PROPERTY CXX_STANDARD 14)
