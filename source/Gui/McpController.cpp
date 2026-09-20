#include "McpController.h"

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
    auto constexpr ServerName = "alien";
    auto constexpr TaskPollInterval = std::chrono::milliseconds(50);
    auto constexpr ServerStoppedMessage = "The MCP server has been stopped.";
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

void McpController::init()
{
    _port = GlobalSettings::get().getValue("settings.mcp server.port", DefaultPort);
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

    auto url = std::format("http://127.0.0.1:{}/mcp", _port);
    log(Priority::Important, "mcp: server started at " + url);
    printOverlayMessage("MCP server running at " + url);
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
    return {
        McpTool{
            .name = "create_simulation",
            .description = "Replaces the current simulation in ALIEN with a new, empty and paused simulation. The simulation parameters of the current "
                           "simulation are kept. Omitted world dimensions default to the current ones.",
            .inputSchema =
                {
                    {"type", "object"},
                    {"properties",
                     boost::json::object{
                         {"width", boost::json::object{{"type", "integer"}, {"minimum", 1}, {"description", "World width"}}},
                         {"height", boost::json::object{{"type", "integer"}, {"minimum", 1}, {"description", "World height"}}},
                         {"project_name", boost::json::object{{"type", "string"}, {"description", "Project name, generated if omitted"}}},
                     }},
                },
            .handler = [this](boost::json::object const& arguments) { return executeOnMainThread([this, arguments] { return createSimulation(arguments); }); },
        },
        McpTool{
            .name = "run_simulation",
            .description = "Starts the simulation in ALIEN.",
            .inputSchema = {{"type", "object"}, {"properties", boost::json::object{}}},
            .handler = [this](boost::json::object const&) { return executeOnMainThread([this] { return runSimulation(); }); },
        },
        McpTool{
            .name = "pause_simulation",
            .description = "Pauses the simulation in ALIEN.",
            .inputSchema = {{"type", "object"}, {"properties", boost::json::object{}}},
            .handler = [this](boost::json::object const&) { return executeOnMainThread([this] { return pauseSimulation(); }); },
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
