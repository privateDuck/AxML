add_executable(AxMLTest
        main.cpp
)

target_link_libraries(AxMLTest
        PRIVATE
        AxML
)

target_compile_definitions(AxMLTest
        PRIVATE
        AXML_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)