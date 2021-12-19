cmake_minimum_required(VERSION 3.20)
project(hsnr_pac_e5)

set(CMAKE_CXX_STANDARD 23)

find_package(MPI REQUIRED)

add_executable(${PROJECT_NAME}
        mmul.h
        mmul.cpp
        main.cpp
        )

target_link_libraries(${PROJECT_NAME}
        PUBLIC MPI::MPI_CXX
        )
