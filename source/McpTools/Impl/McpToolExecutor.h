#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <boost/json/object.hpp>

#include <Network/McpServer.h>

class McpToolExecutor
{
public:
    using ResultFunc = std::function<void(std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result)>;
    McpToolExecutor(ResultFunc const& resultFunc);

    std::vector<McpTool> createToolsExecutedInProcess(std::vector<McpTool> const& tools);

    void process();
    void waitForToolCalls(std::chrono::milliseconds const& maxDuration);

    void beginShutdown();
    void endShutdown();

private:
    McpTool createToolExecutedInProcess(McpTool const& tool);
    McpToolResult executeInProcess(std::function<void(McpToolCompletion const&)> const& function);

    ResultFunc _resultFunc;

    std::mutex _pendingTasksMutex;
    std::condition_variable _taskAdded;
    std::vector<std::function<void()>> _pendingTasks;
    std::atomic<bool> _shuttingDown = false;
};
