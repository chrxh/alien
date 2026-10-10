#include "McpController.h"

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <format>
#include <future>

#include <boost/json.hpp>

#include <Base/Interface/GlobalSettings.h>
#include <Base/Interface/LoggingService.h>
#include <Base/Interface/Resources.h>
#include <Base/Interface/StringHelper.h>

#include <Engine/Interface/SimulationFacade.h>

#include <McpTools/Interface/McpToolsFacade.h>

#include "BrowserWindow.h"
#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "MainLoopController.h"
#include "NewSimulationService.h"
#include "OverlayController.h"
#include "PictureGuiService.h"
#include "SimulationView.h"
#include "Viewport.h"

namespace
{
    auto constexpr DefaultPort = 8765;
    auto constexpr MinPort = 1;
    auto constexpr MaxPort = 65535;
    auto constexpr ServerName = "alien";
    auto constexpr TaskPollInterval = std::chrono::milliseconds(50);
    auto constexpr ServerStoppedMessage = "The MCP server has been stopped.";
    auto constexpr MaxCommandLogEntries = size_t{1000};
    auto constexpr MaxLogTextLength = size_t{1000};
    auto constexpr MaxLogTextLines = size_t{20};
}

McpController::~McpController()
{
    _stopping = true;
    _server.reset();
}

bool McpController::isServerRunning() const
{
    return _server != nullptr;
}

void McpController::setServerRunning(bool value)
{
    if (value) {
        startServer();
        if (isServerRunning()) {
            printOverlayMessage("MCP server running at " + getServerUrl());
        }
    } else if (isServerRunning()) {
        stopServer();
        printOverlayMessage("MCP server stopped");
    }
}

int McpController::getPort() const
{
    return _port;
}

int McpController::getDefaultPort() const
{
    return DefaultPort;
}

void McpController::setPort(int value)
{
    auto port = std::clamp(value, MinPort, MaxPort);
    if (port == _port) {
        return;
    }
    _port = port;
    if (isServerRunning()) {
        stopServer();
        setServerRunning(true);
    }
}

std::string McpController::getServerUrl() const
{
    return std::format("http://127.0.0.1:{}/mcp", _port);
}

std::vector<McpToolGroup> const& McpController::getToolGroups() const
{
    return _toolGroups;
}

std::deque<McpCommandLogEntry> const& McpController::getCommandLog() const
{
    return _commandLog;
}

void McpController::clearCommandLog()
{
    _commandLog.clear();
}

void McpController::init()
{
    for (auto const& tool : createTools()) {
        if (_toolGroups.empty() || _toolGroups.back().name != tool.group) {
            _toolGroups.emplace_back(McpToolGroup{.name = tool.group});
        }
        _toolGroups.back().toolNames.emplace_back(tool.name);
    }
    setPort(GlobalSettings::get().getValue("settings.mcp server.port", DefaultPort));
    if (GlobalSettings::get().getValue("settings.mcp server.enabled", false)) {
        startServer();
    }
}

void McpController::process()
{
    if (!MainLoopController::get().isOperatingMode()) {
        return;
    }
    _McpToolsFacade::get()->process();

    decltype(_pendingTasks) tasks;
    {
        std::lock_guard lock(_pendingTasksMutex);
        tasks.swap(_pendingTasks);
    }
    for (auto const& task : tasks) {
        task();
    }
}

void McpController::shutdown()
{
    GlobalSettings::get().setValue("settings.mcp server.port", _port);
    GlobalSettings::get().setValue("settings.mcp server.enabled", isServerRunning());
    stopServer();
}

RealVector2D McpController::getVisibleAreaCenter() const
{
    return Viewport::get().getCenterInWorldPos();
}

RealVector2D McpController::getVisibleAreaSize() const
{
    auto viewSize = Viewport::get().getViewSize();
    auto zoom = Viewport::get().getZoomFactor();
    return {toFloat(viewSize.x) / zoom, toFloat(viewSize.y) / zoom};
}

float McpController::getZoomFactor() const
{
    return Viewport::get().getZoomFactor();
}

void McpController::setVisibleArea(RealVector2D const& center, float zoomFactor)
{
    Viewport::get().setCenterInWorldPos(center);
    Viewport::get().setZoomFactor(zoomFactor);
}

void McpController::createSimulation(std::string const& projectName, IntVector2D const& worldSize)
{
    NewSimulationService::get().createSimulation(NewSimulationService::Parameters()
                                                     .projectName(projectName)
                                                     .worldSize(worldSize)
                                                     .externalEnergy(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
}

void McpController::applySimulation(SimulationDesc const& simulation)
{
    if (auto errorMessage = NewSimulationService::get().loadSimulation(simulation)) {
        throw std::runtime_error(*errorMessage);
    }
}

void McpController::onSelectionChanged()
{
    EditorModel::get().update();
}

void McpController::onNetworkResourcesChanged()
{
    BrowserWindow::get().onRefresh();
}

std::string McpController::createPicture(IntVector2D const& resolution, McpPictureFormat format)
{
    auto picture = SimulationView::get().savePicture(resolution);
    return format == McpPictureFormat::Png ? PictureGuiService::get().encodePng(picture) : PictureGuiService::get().encodeJpg(picture);
}

std::optional<std::string> McpController::createSimulationPreviewJpg()
{
    return PictureGuiService::get().createSimulationPreviewJpg();
}

void McpController::showMessage(std::string const& message)
{
    printOverlayMessage(message);
}

void McpController::startServer()
{
    if (_server) {
        return;
    }
    auto server = std::make_unique<McpServer>(ServerName, Const::ProgramVersion, createTools());
    if (!server->start(_port)) {
        log(Priority::Important, std::format("mcp: could not start server on port {}", _port));
        GenericMessageDialog::get().information("MCP server", std::format("The MCP server could not be started because port {} is not available.", _port));
        return;
    }
    _server = std::move(server);

    log(Priority::Important, "mcp: server started at " + getServerUrl());
}

void McpController::stopServer()
{
    if (!_server) {
        return;
    }
    _stopping = true;
    _server->stop();
    _server.reset();
    {
        std::lock_guard lock(_pendingTasksMutex);
        _pendingTasks.clear();
    }
    _stopping = false;

    log(Priority::Important, "mcp: server stopped");
}

std::vector<McpTool> McpController::createTools()
{
    std::vector<McpTool> result;
    for (auto const& tool : _McpToolsFacade::get()->getTools(*this)) {
        result.emplace_back(wrapTool(tool));
    }
    return result;
}

McpTool McpController::wrapTool(McpTool const& tool)
{
    auto result = tool;
    result.deferredHandler = nullptr;
    result.handler = [this, tool](boost::json::object const& arguments) {
        return executeOnMainThread([this, tool, arguments](McpToolCompletion const& completion) {
            auto completeAndLog = [this, name = tool.name, arguments, completion](McpToolResult const& toolResult) {
                addCommandLogEntry(name, arguments, toolResult);
                completion(toolResult);
            };
            try {
                if (tool.handler) {
                    completeAndLog(tool.handler(arguments));
                } else {
                    tool.deferredHandler(arguments, completeAndLog);
                }
            } catch (std::exception const& exception) {
                completeAndLog(McpToolResult{.text = exception.what(), .isError = true});
            }
        });
    };
    return result;
}

McpToolResult McpController::executeOnMainThread(std::function<void(McpToolCompletion const&)> const& function)
{
    auto promise = std::make_shared<std::promise<McpToolResult>>();
    auto completed = std::make_shared<std::atomic_flag>();
    auto result = promise->get_future();
    {
        std::lock_guard lock(_pendingTasksMutex);
        if (_stopping) {
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
    while (result.wait_for(TaskPollInterval) != std::future_status::ready) {
        if (_stopping) {
            return McpToolResult{.text = ServerStoppedMessage, .isError = true};
        }
    }
    return result.get();
}

void McpController::addCommandLogEntry(std::string const& toolName, boost::json::object const& arguments, McpToolResult const& result)
{
    auto command = arguments.empty() ? toolName : toolName + " " + boost::json::serialize(arguments);
    log(Priority::Important, result.isError ? std::format("mcp: {} -> error: {}", toolName, result.text) : "mcp: " + toolName);

    _commandLog.emplace_back(McpCommandLogEntry{
        .time = std::chrono::system_clock::now(),
        .command = StringHelper::truncate(command, MaxLogTextLength, MaxLogTextLines),
        .result = StringHelper::truncate(result.text, MaxLogTextLength, MaxLogTextLines),
        .isError = result.isError});
    if (_commandLog.size() > MaxCommandLogEntries) {
        _commandLog.pop_front();
    }
}
