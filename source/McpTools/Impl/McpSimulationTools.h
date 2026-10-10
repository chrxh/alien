#pragma once

#include <vector>

#include <Base/Interface/Singleton.h>

#include <Network/Interface/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

#include "McpPersisterTask.h"

class McpSimulationTools
{
    MAKE_SINGLETON(McpSimulationTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);
    void process();

private:
    McpToolResult createSimulation(boost::json::object const& arguments) const;
    McpToolResult resizeWorld(boost::json::object const& arguments) const;
    void saveSimulation(boost::json::object const& arguments, McpToolCompletion const& completion);
    void loadSimulation(boost::json::object const& arguments, McpToolCompletion const& completion);

    McpToolContext* _context = nullptr;
    McpPersisterTask _saveTask;
    McpPersisterTask _loadTask;
};
