#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <future>

#include <Base/Singleton.h>

#include <Network/McpServer.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

class McpController : public MainLoopEntity
{
    MAKE_SINGLETON(McpController);

public:
    bool isServerRunning() const;
    void setServerRunning(bool value);

private:
    void init() override;
    void process() override;
    void shutdown() override;

    void startServer();
    void stopServer();

    std::vector<McpTool> createTools();
    McpToolResult executeOnMainThread(std::function<McpToolResult()> const& function);

    McpToolResult createSimulation(boost::json::object const& arguments);
    McpToolResult runSimulation();
    McpToolResult pauseSimulation();

    int _port = 0;
    std::unique_ptr<McpServer> _server;

    std::mutex _pendingTasksMutex;
    std::vector<std::shared_ptr<std::packaged_task<McpToolResult()>>> _pendingTasks;
    std::atomic<bool> _stopping = false;
};
