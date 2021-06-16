project(Base)

set(Base_SOURCES
    BaseServices.cpp
    BaseServices.h
    DebugMacros.h
    Definitions.cpp
    Definitions.h
    DllExport.h
    Exceptions.h
    GlobalFactory.h
    GlobalFactoryImpl.cpp
    GlobalFactoryImpl.h
    Job.cpp
    Job.h
    LoggingService.h
    LoggingServiceImpl.cpp
    LoggingServiceImpl.h
    NumberGenerator.h
    NumberGeneratorImpl.cpp
    NumberGeneratorImpl.h
    ServiceLocator.cpp
    ServiceLocator.h
    Tracker.h
    Worker.cpp
    Worker.h
)

add_library(Base SHARED ${Base_SOURCES})
add_library(ALiEn::Base ALIAS Base)

target_compile_definitions(Base PRIVATE BASE_LIB)

target_link_libraries(Base PUBLIC Qt6::Widgets Boost::headers)

target_include_directories(Base
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
