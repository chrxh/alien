#include "McpController.h"

#include <algorithm>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>

#include <EngineInterface/SimulationFacade.h>

#include <McpToolsInterface/McpToolsFacade.h>

#include "EditorModel.h"
#include "GenericMessageDialog.h"
#include "MainLoopController.h"
#include "NewSimulationService.h"
#include "OverlayController.h"
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
    _port = std::clamp(value, MinPort, MaxPort);
}

std::string McpController::getServerUrl() const
{
    return std::format("http://127.0.0.1:{}/mcp", _port);
}

std::vector<std::string> const& McpController::getToolNames() const
{
    return _toolNames;
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
        _toolNames.emplace_back(tool.name);
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
    decltype(_pendingTasks) tasks;
    {
        std::lock_guard lock(_pendingTasksMutex);
        tasks.swap(_pendingTasks);
    }
    for (auto const& task : tasks) {
        (*task)();
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

void McpController::createSimulation(std::string const& projectName, IntVector2D const& worldSize)
{
    NewSimulationService::get().createSimulation(NewSimulationService::Parameters()
                                                     .projectName(projectName)
                                                     .worldSize(worldSize)
                                                     .externalEnergy(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
}

void McpController::onSelectionChanged()
{
    EditorModel::get().update();
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
    result.handler = [this, name = tool.name, handler = tool.handler](boost::json::object const& arguments) {
        return executeOnMainThread([this, name, handler, arguments] {
            auto toolResult = [&] {
                try {
                    return handler(arguments);
                } catch (std::exception const& exception) {
                    return McpToolResult{.text = exception.what(), .isError = true};
                }
            }();
            addCommandLogEntry(name, arguments, toolResult);
            return toolResult;
        });
    };
    return result;
}

McpToolResult McpController::executeOnMainThread(std::function<McpToolResult()> const& function)
{
    auto task = std::make_shared<std::packaged_task<McpToolResult()>>(function);
    auto result = task->get_future();
    {
        std::lock_guard lock(_pendingTasksMutex);
        if (_stopping) {
            return McpToolResult{.text = ServerStoppedMessage, .isError = true};
        }
        _pendingTasks.emplace_back(task);
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
    log(Priority::Important, std::format("mcp: {} -> {}{}", command, result.isError ? "error: " : "", result.text));

    _commandLog.emplace_back(
        McpCommandLogEntry{.time = std::chrono::system_clock::now(), .command = command, .result = result.text, .isError = result.isError});
    if (_commandLog.size() > MaxCommandLogEntries) {
        _commandLog.pop_front();
    }
}
