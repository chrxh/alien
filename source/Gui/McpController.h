#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <future>

#include <Base/Singleton.h>

#include <Network/McpServer.h>
#include <McpTools/McpToolContext.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

struct McpCommandLogEntry
{
    std::chrono::system_clock::time_point time;
    std::string command;
    std::string result;
    bool isError = false;
};

class McpController
    : public MainLoopEntity
    , public McpToolContext
{
    MAKE_SINGLETON(McpController);

public:
    bool isServerRunning() const;
    void setServerRunning(bool value);

    int getPort() const;
    int getDefaultPort() const;
    void setPort(int value);
    std::string getServerUrl() const;
    std::vector<std::string> const& getToolNames() const;

    std::deque<McpCommandLogEntry> const& getCommandLog() const;
    void clearCommandLog();

private:
    void init() override;
    void process() override;
    void shutdown() override;

    RealVector2D getVisibleAreaCenter() const override;
    RealVector2D getVisibleAreaSize() const override;
    void createSimulation(std::string const& projectName, IntVector2D const& worldSize) override;
    void onSelectionChanged() override;
    std::optional<RgbImage> loadImage(std::filesystem::path const& path) const override;
    void showMessage(std::string const& message) override;

    void startServer();
    void stopServer();

    std::vector<McpTool> createTools();
    McpTool wrapTool(McpTool const& tool);
    McpToolResult executeOnMainThread(std::function<McpToolResult()> const& function);
    void addCommandLogEntry(std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result);

    int _port = 0;
    std::unique_ptr<McpServer> _server;
    std::vector<std::string> _toolNames;
    std::deque<McpCommandLogEntry> _commandLog;

    std::mutex _pendingTasksMutex;
    std::vector<std::shared_ptr<std::packaged_task<McpToolResult()>>> _pendingTasks;
    std::atomic<bool> _stopping = false;
};
