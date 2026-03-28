cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)

project(tensor LANGUAGES CXX)

set(SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/tensor.cpp
    ${CMAKE_CURRENT_LIST_DIR}/tensor_bool.cpp
)

set(HEADERS
    ${CMAKE_CURRENT_LIST_DIR}/tensor.h
    ${CMAKE_CURRENT_LIST_DIR}/random_generator.h
)

add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
