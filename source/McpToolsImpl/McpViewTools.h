#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Network/McpServer.h>
#include <McpToolsInterface/McpToolContext.h>

class McpViewTools
{
    MAKE_SINGLETON(McpViewTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult getSimulationInfo() const;
    McpToolResult getStatistics(boost::json::object const& arguments) const;
    McpToolResult setView(boost::json::object const& arguments) const;
    McpToolResult takeScreenshot(boost::json::object const& arguments) const;

    void applyView(boost::json::object const& arguments) const;
    boost::json::object describeVisibleArea() const;

    McpToolContext* _context = nullptr;
};
