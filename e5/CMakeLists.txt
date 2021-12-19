cmake_minimum_required(VERSION 3.16.3)
project(hsnr_pac_e5)

set(CMAKE_CXX_STANDARD 21)

find_package(MPI REQUIRED)

add_executable(${PROJECT_NAME}
        mmul.h
        mmul.cpp
        main.cpp
        )

target_link_libraries(${PROJECT_NAME}
        PUBLIC MPI::MPI_CXX
        )
