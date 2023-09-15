cmake_minimum_required(VERSION 3.8)
project(Orient2D)

set(CMAKE_CXX_STANDARD 11)

#file(GLOB *.h *.cpp)
aux_source_directory(${PROJECT_SOURCE_DIR}/src src_dirs)

# 条件判断
if(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall -O0")
    message("Debug mode:${CMAKE_C_FLAGS_DEBUG}")
    # add_executable(test_debug ${src_dirs})

elseif(CMAKE_BUILD_TYPE AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Wall -O3")
    message("Release mode:${CMAKE_C_FLAGS_RELEASE}")
    # add_executable(test_release ${src_dirs})
else()
    message("else:${CMAKE_BUILD_TYPE}")
    message("else:${CMAKE_C_FLAGS_RELEASE}")
    # add_executable(test_release ${src_dirs})
endif()

set(SOURCE_FILES main.cpp BSpline.cpp)
set(SOURCE_FILES1 pyBsplineInterp.cpp BSpline.cpp)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
find_package(Eigen3 REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(pybind11_DIR ~/miniconda3/envs/face1/lib/python3.8/site-packages/pybind11/share/cmake/pybind11)
FIND_PACKAGE(pybind11 REQUIRED)

INCLUDE_DIRECTORIES(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${pybind11_INCLUDE_DIRS}
        )

# add_executable(BsplineInterpTest ${SOURCE_FILES} )
pybind11_add_module(pyBsplineInterp ${SOURCE_FILES1})


# add_library(pyBsplineInterp ${SOURCE_FILES1})
# add_subdirectory(pybind11)


# target_link_libraries(pyBsplineInterp PRIVATE 
# # Eigen3::Eigen
# # ${CMAKE_CURRENT_SOURCE_DIR}/../_deps/libOpenMeshCore.so
# # ${CMAKE_CURRENT_SOURCE_DIR}/../_deps/libOpenMeshTools.so
# )