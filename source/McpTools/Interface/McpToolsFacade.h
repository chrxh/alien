#pragma once

#include <vector>

#include <Network/Interface/McpServer.h>

#include "Definitions.h"
#include "McpToolContext.h"

class _McpToolsFacade
{
public:
    virtual ~_McpToolsFacade() = default;

    static McpToolsFacade get();

    virtual std::vector<McpTool> getTools(McpToolContext& context) = 0;
    virtual void process() = 0;

protected:
    static McpToolsFacade _instance;
};
