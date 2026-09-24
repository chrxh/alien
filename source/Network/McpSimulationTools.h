#pragma once

#include <vector>

#include <Base/Singleton.h>

#include "McpHost.h"
#include "McpServer.h"

class McpSimulationTools
{
    MAKE_SINGLETON(McpSimulationTools);

public:
    std::vector<McpTool> getTools(McpHost const& host);

private:
    McpToolResult getSimulationInfo() const;
    McpToolResult createSimulation(boost::json::object const& arguments) const;
    McpToolResult runSimulation() const;
    McpToolResult pauseSimulation() const;

    McpHost _host;
};
