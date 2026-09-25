#pragma once

#include <vector>

#include <Base/Singleton.h>

#include <Network/McpServer.h>

#include "McpToolContext.h"

class McpToolsService
{
    MAKE_SINGLETON(McpToolsService);

public:
    std::vector<McpTool> getTools(McpToolContext& context);
};
