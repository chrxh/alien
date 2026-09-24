
add_library(Network
    Definitions.h
    McpArguments.cpp
    McpArguments.h
    McpCreatorTools.cpp
    McpCreatorTools.h
    McpHost.h
    McpMultiplierTools.cpp
    McpMultiplierTools.h
    McpParameterTools.cpp
    McpParameterTools.h
    McpSchema.cpp
    McpSchema.h
    McpSelectionTools.cpp
    McpSelectionTools.h
    McpServer.cpp
    McpServer.h
    McpService.cpp
    McpService.h
    McpSimulationTools.cpp
    McpSimulationTools.h
    NetworkService.cpp
    NetworkService.h
    NetworkResourceParserService.cpp
    NetworkResourceParserService.h
    NetworkResourceRawTO.cpp
    NetworkResourceRawTO.h
    NetworkResourceService.cpp
    NetworkResourceService.h
    NetworkResourceTreeTO.cpp
    NetworkResourceTreeTO.h
    NetworkValidationService.cpp
    NetworkValidationService.h
    UserTO.h)

target_link_libraries(Network Base)
target_link_libraries(Network EngineInterface)
target_link_libraries(Network PersisterInterface)
target_link_libraries(Network Boost::boost)
target_link_libraries(Network Boost::json)
target_link_libraries(Network OpenSSL::SSL OpenSSL::Crypto)
    
target_precompile_headers(Network PRIVATE
    <memory>
    <optional>
    <string>
    <vector>
)

if (MSVC)
    target_compile_options(Network PRIVATE "/MP")
endif()
