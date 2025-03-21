set(SOURCES
    data_reader.cpp
    model.cpp
    activation.cpp
    loss.cpp
    optimizer.cpp
    metric.cpp
    trainer.cpp
    models/lr.cpp
    models/fm.cpp
    utils/config_parser.cpp
    utils/math_util.cpp
)

add_library(simpleflow STATIC ${SOURCES})
target_link_libraries(simpleflow ${BLAS_LIBRARIES}) 