cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)

project(sample LANGUAGES CXX)

set(SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC tensor)

install(TARGETS ${PROJECT_NAME}
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
)
