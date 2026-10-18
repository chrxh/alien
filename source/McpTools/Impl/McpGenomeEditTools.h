#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Data/Descs.h>

#include <Network/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

class McpGenomeEditTools
{
    MAKE_SINGLETON(McpGenomeEditTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult editGenomes(boost::json::object const& arguments) const;
    McpToolResult summarizeGenomes(boost::json::object const& arguments) const;

    ContentDesc getContent(bool restrictToSelection) const;
    void replaceSelection(ContentDesc&& content) const;
    void replaceWorldContent(ContentDesc const& content) const;

    McpToolContext* _context = nullptr;
};
