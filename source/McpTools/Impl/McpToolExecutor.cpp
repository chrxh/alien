#include "McpToolExecutor.h"

#include <memory>
#include <future>

namespace
{
    auto constexpr ResultPollInterval = std::chrono::milliseconds(50);
    auto constexpr ServerStoppedMessage = "The MCP server has been stopped.";
}

McpToolExecutor::McpToolExecutor(ResultFunc const& resultFunc)
    : _resultFunc(resultFunc)
{}

std::vector<McpTool> McpToolExecutor::createToolsExecutedInProcess(std::vector<McpTool> const& tools)
{
    std::vector<McpTool> result;
    for (auto const& tool : tools) {
        result.emplace_back(createToolExecutedInProcess(tool));
    }
    return result;
}

void McpToolExecutor::process()
{
    decltype(_pendingTasks) tasks;
    {
        std::lock_guard lock(_pendingTasksMutex);
        tasks.swap(_pendingTasks);
    }
    for (auto const& task : tasks) {
        task();
    }
}

void McpToolExecutor::waitForToolCalls(std::chrono::milliseconds const& maxDuration)
{
    std::unique_lock lock(_pendingTasksMutex);
    _taskAdded.wait_for(lock, maxDuration, [this] { return !_pendingTasks.empty(); });
}

void McpToolExecutor::beginShutdown()
{
    _shuttingDown = true;
}

void McpToolExecutor::endShutdown()
{
    std::lock_guard lock(_pendingTasksMutex);
    _pendingTasks.clear();
    _shuttingDown = false;
}

McpTool McpToolExecutor::createToolExecutedInProcess(McpTool const& tool)
{
    auto result = tool;
    result.deferredHandler = nullptr;
    result.handler = [this, tool](boost::json::object const& arguments) {
        return executeInProcess([this, tool, arguments](McpToolCompletion const& completion) {
            auto reportAndComplete = [this, name = tool.name, arguments, completion](McpToolResult const& toolResult) {
                _resultFunc(name, arguments, toolResult);
                completion(toolResult);
            };
            try {
                if (tool.handler) {
                    reportAndComplete(tool.handler(arguments));
                } else {
                    tool.deferredHandler(arguments, reportAndComplete);
                }
            } catch (std::exception const& exception) {
                reportAndComplete(McpToolResult{.text = exception.what(), .isError = true});
            }
        });
    };
    return result;
}

McpToolResult McpToolExecutor::executeInProcess(std::function<void(McpToolCompletion const&)> const& function)
{
    auto promise = std::make_shared<std::promise<McpToolResult>>();
    auto completed = std::make_shared<std::atomic_flag>();
    auto result = promise->get_future();
    {
        std::lock_guard lock(_pendingTasksMutex);
        if (_shuttingDown) {
            return McpToolResult{.text = ServerStoppedMessage, .isError = true};
        }
        _pendingTasks.emplace_back([function, promise, completed] {
            function([promise, completed](McpToolResult const& toolResult) {
                if (!completed->test_and_set()) {
                    promise->set_value(toolResult);
                }
            });
        });
    }
    _taskAdded.notify_one();

    while (result.wait_for(ResultPollInterval) != std::future_status::ready) {
        if (_shuttingDown) {
            return McpToolResult{.text = ServerStoppedMessage, .isError = true};
        }
    }
    return result.get();
}
