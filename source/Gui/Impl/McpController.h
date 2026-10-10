#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <Base/Interface/Singleton.h>

#include <Network/Interface/McpServer.h>
#include <McpTools/Interface/McpToolContext.h>

#include "Definitions.h"
#include "MainLoopEntity.h"

struct McpCommandLogEntry
{
    std::chrono::system_clock::time_point time;
    std::string command;
    std::string result;
    bool isError = false;
};

struct McpToolGroup
{
    std::string name;
    std::vector<std::string> toolNames;
};

class McpController
    : public MainLoopEntity
    , public McpToolContext
{
    MAKE_SINGLETON(McpController);

public:
    ~McpController() override;

    bool isServerRunning() const;
    void setServerRunning(bool value);

    int getPort() const;
    int getDefaultPort() const;
    void setPort(int value);
    std::string getServerUrl() const;
    std::vector<McpToolGroup> const& getToolGroups() const;

    std::deque<McpCommandLogEntry> const& getCommandLog() const;
    void clearCommandLog();

private:
    void init() override;
    void process() override;
    void shutdown() override;

    RealVector2D getVisibleAreaCenter() const override;
    RealVector2D getVisibleAreaSize() const override;
    float getZoomFactor() const override;
    void setVisibleArea(RealVector2D const& center, float zoomFactor) override;
    void createSimulation(std::string const& projectName, IntVector2D const& worldSize) override;
    void applySimulation(SimulationDesc const& simulation) override;
    void onSelectionChanged() override;
    void onNetworkResourcesChanged() override;
    std::string createPicture(IntVector2D const& resolution, McpPictureFormat format) override;
    std::optional<std::string> createSimulationPreviewJpg() override;
    void showMessage(std::string const& message) override;

    void startServer();
    void stopServer();

    std::vector<McpTool> createTools();
    McpTool wrapTool(McpTool const& tool);
    McpToolResult executeOnMainThread(std::function<void(McpToolCompletion const&)> const& function);
    void addCommandLogEntry(std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result);

    int _port = 0;
    std::unique_ptr<McpServer> _server;
    std::vector<McpToolGroup> _toolGroups;
    std::deque<McpCommandLogEntry> _commandLog;

    std::mutex _pendingTasksMutex;
    std::vector<std::function<void()>> _pendingTasks;
    std::atomic<bool> _stopping = false;
};
