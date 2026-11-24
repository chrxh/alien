#pragma once

#include <memory>

#include <boost/json/object.hpp>

#include <Network/McpServer.h>

#include <McpTools/Interface/McpToolsFacade.h>

#include "McpToolExecutor.h"

class _McpToolsFacadeImpl : public _McpToolsFacade
{
public:
    static void set(McpToolsFacade const& instance);

    _McpToolsFacadeImpl();
    ~_McpToolsFacadeImpl() override;

    void setup(McpToolContext& context) override;
    void shutdown() override;

    bool startServer(int port) override;
    void stopServer() override;
    bool isServerRunning() const override;

    void processToolCalls() override;
    void waitForToolCalls(std::chrono::milliseconds const& maxDuration) override;

    std::vector<McpToolGroup> const& getToolGroups() const override;
    std::deque<McpCommandLogEntry> const& getCommandLog() const override;
    void clearCommandLog() override;

private:
    std::vector<McpTool> createTools(McpToolContext& context) const;
    void addCommandLogEntry(std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result);

    McpToolContext* _context = nullptr;
    std::vector<McpTool> _tools;
    std::vector<McpToolGroup> _toolGroups;
    std::deque<McpCommandLogEntry> _commandLog;

    McpToolExecutor _executor;
    std::unique_ptr<McpServer> _server;
};
