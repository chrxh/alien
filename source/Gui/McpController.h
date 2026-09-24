#pragma once

#include <Base/Singleton.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

// Runs the MCP service of the Network component inside ALIEN
class McpController : public MainLoopEntity
{
    MAKE_SINGLETON(McpController);

private:
    void init() override;
    void process() override;
    void shutdown() override;
};
