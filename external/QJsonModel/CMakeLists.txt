project(QJsonModel)

set(QJsonModel_SOURCES
    qjsonmodel.cpp
    qjsonmodel.h
)

add_library(QJsonModel SHARED ${QJsonModel_SOURCES})

target_link_libraries(QJsonModel PUBLIC Qt6::Widgets)

target_include_directories(QJsonModel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
