find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(TEST_SOURCES
    activation_test.cpp
    loss_test.cpp
    data_reader_test.cpp
    lr_test.cpp
    fm_test.cpp
    metric_test.cpp
    optimizer_test.cpp
    model_test.cpp
    utils_test.cpp
    trainer_test.cpp
    main_test.cpp
)

add_executable(unit_tests ${TEST_SOURCES})
target_link_libraries(unit_tests simpleflow ${GTEST_LIBRARIES} pthread) 