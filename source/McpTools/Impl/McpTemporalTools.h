#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <Base/Interface/Singleton.h>

#include <Network/Interface/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

class McpTemporalTools
{
    MAKE_SINGLETON(McpTemporalTools);

public:
    std::vector<McpTool> getTools(McpToolContext& context);
    void process();

private:
    McpToolResult getTimeInfo() const;
    McpToolResult runSimulation() const;
    McpToolResult pauseSimulation() const;
    void calcTimesteps(boost::json::object const& arguments, McpToolCompletion const& completion);
    McpToolResult restorePreviousTimestep() const;
    McpToolResult createFlashback() const;
    McpToolResult restoreFlashback() const;
    McpToolResult setSpeedLimit(boost::json::object const& arguments) const;

    void checkPaused() const;
    std::string describeTime() const;

    McpToolContext* _context = nullptr;

    struct PendingTimesteps
    {
        uint64_t numRemainingTimesteps = 0;
        uint64_t numTimesteps = 0;
        int sessionId = 0;
        McpToolCompletion completion;
    };
    std::optional<PendingTimesteps> _pendingTimesteps;
};
