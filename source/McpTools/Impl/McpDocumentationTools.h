#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Network/McpServer.h>

class McpDocumentationTools
{
    MAKE_SINGLETON(McpDocumentationTools);

public:
    std::vector<McpTool> getTools();

private:
    McpToolResult readDocumentation(boost::json::object const& arguments) const;
};
