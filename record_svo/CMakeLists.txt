cmake_minimum_required(VERSION 3.5)
PROJECT(ZED_face_privacy)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_BUILD_TYPE "Release")

find_package(ZED REQUIRED)
find_package(CUDA ${ZED_CUDA_VERSION} REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})

IF (CMAKE_SYSTEM_PROCESSOR MATCHES aarch64)
    add_definitions(-DJETSON_STYLE)
ENDIF()

FILE(GLOB_RECURSE SRC_FILES src/*.c*)

ADD_EXECUTABLE(${PROJECT_NAME}  ${SRC_FILES})
SET(ZED_LIBS ${ZED_LIBRARIES} ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY})

target_link_libraries(${PROJECT_NAME}
                        ${ZED_LIBS}
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARIES}
                        ${GLEW_LIBRARIES}
                        ${OpenCV_LIBRARIES})

