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

#include "McpHost.h"
#include "McpServer.h"

struct McpCommandLogEntry
{
    std::chrono::system_clock::time_point time;
    std::string command;
    std::string result;
    bool isError = false;
};

class McpService
{
    MAKE_SINGLETON(McpService);

public:
    // Restores the settings and restarts the server if it was running at the last shutdown
    void init(McpHost const& host);
    void shutdown();

    // Executes the pending tool calls, has to be called regularly on the main thread
    void processPendingCommands();

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
    void startServer();
    void stopServer();

    std::vector<McpTool> createTools();
    McpTool wrapTool(McpTool const& tool);
    McpToolResult executeOnMainThread(std::function<McpToolResult()> const& function);
    void addCommandLogEntry(std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result);

    McpHost _host;
    int _port = 0;
    std::unique_ptr<McpServer> _server;
    std::vector<std::string> _toolNames;
    std::deque<McpCommandLogEntry> _commandLog;

    std::mutex _pendingTasksMutex;
    std::vector<std::shared_ptr<std::packaged_task<McpToolResult()>>> _pendingTasks;
    std::atomic<bool> _stopping = false;
};
