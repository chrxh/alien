#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Network/McpServer.h>

#include "McpToolContext.h"

class McpSimulationTools
{
    MAKE_SINGLETON(McpSimulationTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);

private:
    McpToolResult getSimulationInfo() const;
    McpToolResult createSimulation(boost::json::object const& arguments) const;
    McpToolResult runSimulation() const;
    McpToolResult pauseSimulation() const;

    McpToolContext* _context = nullptr;
};
