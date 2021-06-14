project(Web)

set(Web_SOURCES
    Definitions.h
    DllExport.h
    HttpClient.cpp
    HttpClient.h
    Parser.cpp
    Parser.h
    SimulationInfo.h
    Task.h
    WebAccess.h
    WebAccessImpl.cpp
    WebAccessImpl.h
    WebBuilderFacade.h
    WebBuilderFacadeImpl.cpp
    WebBuilderFacadeImpl.h
    WebServices.cpp
    WebServices.h
)

add_library(Web SHARED ${Web_SOURCES})
add_library(ALiEn::Web ALIAS Web)

target_compile_definitions(Web PRIVATE BASE_LIB)

target_link_libraries(Web PUBLIC Qt6::Widgets Qt6::Network)

target_include_directories(Web
PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
