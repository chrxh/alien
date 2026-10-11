add_library(McpToolsInterface
    Definitions.h
    McpToolContext.h
    McpToolsFacade.cpp
    McpToolsFacade.h)

target_link_libraries(McpToolsInterface Base)
target_link_libraries(McpToolsInterface Network)
target_link_libraries(McpToolsInterface Boost::boost)
target_link_libraries(McpToolsInterface Boost::json)

if (MSVC)
    target_compile_options(McpToolsInterface PRIVATE "/MP")
endif()
