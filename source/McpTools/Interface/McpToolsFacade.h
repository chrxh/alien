#pragma once

#include <chrono>
#include <deque>
#include <vector>

#include "Definitions.h"
#include "McpCommandLogEntry.h"
#include "McpToolContext.h"
#include "McpToolGroup.h"

class _McpToolsFacade
{
public:
    virtual ~_McpToolsFacade() = default;

    static McpToolsFacade get();

    virtual void setup(McpToolContext& context) = 0;
    virtual void shutdown() = 0;

    virtual bool startServer(int port) = 0;
    virtual void stopServer() = 0;
    virtual bool isServerRunning() const = 0;

    virtual void processToolCalls() = 0;
    virtual void waitForToolCalls(std::chrono::milliseconds const& maxDuration) = 0;

    virtual std::vector<McpToolGroup> const& getToolGroups() const = 0;
    virtual std::deque<McpCommandLogEntry> const& getCommandLog() const = 0;
    virtual void clearCommandLog() = 0;

protected:
    static McpToolsFacade _instance;
};
