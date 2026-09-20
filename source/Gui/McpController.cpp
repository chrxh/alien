#include "McpController.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <format>

#include <boost/json.hpp>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>

#include <EngineInterface/NameGeneratorService.h>
#include <EngineInterface/SimulationFacade.h>

#include "GenericMessageDialog.h"
#include "MainLoopController.h"
#include "NewSimulationService.h"
#include "OverlayController.h"

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
    } else {
        stopServer();
    }
}

int McpController::getPort() const
{
    return _port;
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
    printOverlayMessage("MCP server running at " + getServerUrl());
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
    printOverlayMessage("MCP server stopped");
}

std::vector<McpTool> McpController::createTools()
{
    auto const noArguments = boost::json::object{{"type", "object"}, {"properties", boost::json::object{}}};
    return {
        createTool(
            "create_simulation",
            "Replaces the current simulation in ALIEN with a new, empty and paused simulation. The simulation parameters of the current simulation are kept. "
            "Omitted world dimensions default to the current ones.",
            {
                {"type", "object"},
                {"properties",
                 boost::json::object{
                     {"width", boost::json::object{{"type", "integer"}, {"minimum", 1}, {"description", "World width"}}},
                     {"height", boost::json::object{{"type", "integer"}, {"minimum", 1}, {"description", "World height"}}},
                     {"project_name", boost::json::object{{"type", "string"}, {"description", "Project name, generated if omitted"}}},
                 }},
            },
            [this](boost::json::object const& arguments) { return createSimulation(arguments); }),
        createTool("run_simulation", "Starts the simulation in ALIEN.", noArguments, [this](boost::json::object const&) { return runSimulation(); }),
        createTool("pause_simulation", "Pauses the simulation in ALIEN.", noArguments, [this](boost::json::object const&) { return pauseSimulation(); }),
    };
}

McpTool McpController::createTool(
    std::string const& name,
    std::string const& description,
    boost::json::object const& inputSchema,
    std::function<McpToolResult(boost::json::object const&)> const& function)
{
    return McpTool{
        .name = name,
        .description = description,
        .inputSchema = inputSchema,
        .handler =
            [this, name, function](boost::json::object const& arguments) {
                return executeOnMainThread([this, name, function, arguments] {
                    auto result = [&] {
                        try {
                            return function(arguments);
                        } catch (std::exception const& exception) {
                            return McpToolResult{.text = exception.what(), .isError = true};
                        }
                    }();
                    addCommandLogEntry(name, arguments, result);
                    return result;
                });
            },
    };
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

namespace
{
    std::optional<int> getWorldDimension(boost::json::object const& arguments, std::string_view key)
    {
        auto value = arguments.if_contains(key);
        if (!value) {
            return std::nullopt;
        }
        try {
            auto result = value->to_number<int64_t>();
            if (result >= 1 && result <= std::numeric_limits<int>::max()) {
                return static_cast<int>(result);
            }
        } catch (...) {
        }
        throw std::invalid_argument(std::format("'{}' must be a positive integer.", key));
    }
}

McpToolResult McpController::createSimulation(boost::json::object const& arguments)
{
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    worldSize.x = getWorldDimension(arguments, "width").value_or(worldSize.x);
    worldSize.y = getWorldDimension(arguments, "height").value_or(worldSize.y);

    auto projectName = NameGeneratorService::get().createSimulationName();
    if (auto value = arguments.if_contains("project_name")) {
        if (!value->is_string()) {
            throw std::invalid_argument("'project_name' must be a string.");
        }
        projectName = value->as_string();
    }

    NewSimulationService::get().createSimulation(NewSimulationService::Parameters()
                                                     .projectName(projectName)
                                                     .worldSize(worldSize)
                                                     .externalEnergy(_SimulationFacade::get()->getSimulationParameters().externalEnergy.value));
    printOverlayMessage("New simulation");

    return {
        .text = std::format(
            "Created the empty simulation '{}' with a world size of {} x {}. The simulation is {}.",
            projectName,
            worldSize.x,
            worldSize.y,
            _SimulationFacade::get()->isSimulationRunning() ? "running" : "paused")};
}

McpToolResult McpController::runSimulation()
{
    if (_SimulationFacade::get()->isSimulationRunning()) {
        return {.text = "The simulation is already running."};
    }
    _SimulationFacade::get()->runSimulation();
    printOverlayMessage("Run");
    return {.text = "The simulation is running."};
}

McpToolResult McpController::pauseSimulation()
{
    if (!_SimulationFacade::get()->isSimulationRunning()) {
        return {.text = "The simulation is already paused."};
    }
    _SimulationFacade::get()->pauseSimulation();
    printOverlayMessage("Pause");
    return {.text = "The simulation is paused."};
}
