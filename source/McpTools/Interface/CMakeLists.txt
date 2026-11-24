add_library(McpToolsInterface
    Definitions.h
    McpCommandLogEntry.h
    McpToolContext.h
    McpToolGroup.h
    McpToolsFacade.cpp
    McpToolsFacade.h)

target_link_libraries(McpToolsInterface Base)

if (MSVC)
    target_compile_options(McpToolsInterface PRIVATE "/MP")
endif()
