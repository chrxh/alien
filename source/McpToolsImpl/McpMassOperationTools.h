#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Data/Descs.h>

#include <Network/McpServer.h>
#include <McpToolsInterface/McpToolContext.h>

class McpMassOperationTools
{
    MAKE_SINGLETON(McpMassOperationTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult applyMassOperations(boost::json::object const& arguments) const;

    void replaceSelection(ContentDesc&& content) const;
    void replaceWorldContent(ContentDesc const& content) const;

    McpToolContext* _context = nullptr;
};
