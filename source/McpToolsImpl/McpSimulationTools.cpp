#include "McpSimulationTools.h"

#include <filesystem>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/NameGeneratorService.h>

#include <Data/DescEditService.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/TemporalControlService.h>

#include <PersisterInterface/PersisterFacade.h>

#include <Network/McpArguments.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr MaxWorldSize = 10000;
    auto constexpr SimulationFileExtension = ".sim";
}

std::vector<McpTool> McpSimulationTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "create_simulation",
            .description = "Replaces the current simulation in ALIEN with a new, empty and paused simulation. The simulation parameters of the current "
                           "simulation are kept. Omitted world dimensions default to the current ones.",
            .inputSchema = McpSchema::object({
                {"width", McpSchema::integer("World width", 1, MaxWorldSize)},
                {"height", McpSchema::integer("World height", 1, MaxWorldSize)},
                {"project_name", McpSchema::string("Project name, generated if omitted")},
            }),
            .handler = [this](boost::json::object const& arguments) { return createSimulation(arguments); },
        },
        McpTool{
            .name = "resize_world",
            .description = "Changes the world size of the current simulation while keeping its content, time and parameters. Optionally the "
                           "positions of the content are scaled to the new world size.",
            .inputSchema = McpSchema::object(
                {
                    {"width", McpSchema::integer("New world width", 1, MaxWorldSize)},
                    {"height", McpSchema::integer("New world height", 1, MaxWorldSize)},
                    {"scale_content", McpSchema::boolean("Scale the positions of the content to the new world size. Default: false")},
                },
                {"width", "height"}),
            .handler = [this](boost::json::object const& arguments) { return resizeWorld(arguments); },
        },
        McpTool{
            .name = "save_simulation",
            .description = "Saves the current simulation including its parameters and statistics to a simulation file (*.sim). The current view "
                           "position and zoom are stored as well.",
            .inputSchema = McpSchema::object(
                {
                    {"file_path", McpSchema::string("Absolute path of the file, must end with .sim")},
                    {"overwrite", McpSchema::boolean("Overwrite an existing file. Default: false")},
                },
                {"file_path"}),
            .deferredHandler = [this](boost::json::object const& arguments, McpToolCompletion const& completion) { saveSimulation(arguments, completion); },
        },
        McpTool{
            .name = "load_simulation",
            .description = "Replaces the current simulation in ALIEN by a simulation loaded from a simulation file (*.sim).",
            .inputSchema = McpSchema::object({{"file_path", McpSchema::string("Absolute path of the file")}}, {"file_path"}),
            .deferredHandler = [this](boost::json::object const& arguments, McpToolCompletion const& completion) { loadSimulation(arguments, completion); },
        },
    };
}

void McpSimulationTools::process()
{
    _saveTask.process();
    _loadTask.process();
}

McpToolResult McpSimulationTools::createSimulation(boost::json::object const& arguments) const
{
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    worldSize.x = McpArguments::getOptionalInt(arguments, "width", 1, MaxWorldSize).value_or(worldSize.x);
    worldSize.y = McpArguments::getOptionalInt(arguments, "height", 1, MaxWorldSize).value_or(worldSize.y);
    auto projectName = McpArguments::getOptionalString(arguments, "project_name");
    if (!projectName) {
        projectName = NameGeneratorService::get().createSimulationName();
    }

    _context->createSimulation(*projectName, worldSize);
    _context->showMessage("New simulation");

    return {
        .text = std::format(
            "Created the empty simulation '{}' with a world size of {} x {}. The simulation is {}.",
            *projectName,
            worldSize.x,
            worldSize.y,
            _SimulationFacade::get()->isSimulationRunning() ? "running" : "paused")};
}

McpToolResult McpSimulationTools::resizeWorld(boost::json::object const& arguments) const
{
    IntVector2D worldSize{McpArguments::getInt(arguments, "width", 1, MaxWorldSize), McpArguments::getInt(arguments, "height", 1, MaxWorldSize)};
    auto scaleContent = McpArguments::getOptionalBool(arguments, "scale_content").value_or(false);

    auto origWorldSize = _SimulationFacade::get()->getWorldSize();
    auto timestep = _SimulationFacade::get()->getCurrentTimestep();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto content = _SimulationFacade::get()->getSimulationData();
    auto realTime = _SimulationFacade::get()->getRealTime();
    auto statistics = _SimulationFacade::get()->getStatisticsHistory().getCopiedData();
    _SimulationFacade::get()->closeSimulation();

    _SimulationFacade::get()->newSimulation(timestep, worldSize, parameters);
    if (scaleContent) {
        DescEditService::get().scaleContent(content, origWorldSize, worldSize);
    }
    _SimulationFacade::get()->setSimulationData(content);
    _SimulationFacade::get()->setStatisticsHistory(statistics);
    _SimulationFacade::get()->setRealTime(realTime);
    TemporalControlService::get().createFlashback();
    _context->onSelectionChanged();
    _context->showMessage("World resized");

    return {.text = std::format("Resized the world from {} x {} to {} x {}.", origWorldSize.x, origWorldSize.y, worldSize.x, worldSize.y)};
}

void McpSimulationTools::saveSimulation(boost::json::object const& arguments, McpToolCompletion const& completion)
{
    auto filePath = McpArguments::getString(arguments, "file_path");
    auto path = McpArguments::getFilePath(arguments, "file_path");
    if (!filePath.ends_with(SimulationFileExtension)) {
        throw std::invalid_argument(std::format("The file name must end with '{}'.", SimulationFileExtension));
    }
    auto overwrite = McpArguments::getOptionalBool(arguments, "overwrite").value_or(false);
    if (std::filesystem::exists(path) && !overwrite) {
        throw std::invalid_argument(std::format("The file '{}' already exists. Set 'overwrite' to replace it.", filePath));
    }

    auto requestData = SaveSimulationRequestData{.filename = path, .zoom = _context->getZoomFactor(), .center = _context->getVisibleAreaCenter()};
    _saveTask.execute(
        [requestData](SenderId const& senderId) {
            return _PersisterFacade::get()->scheduleSaveSimulation(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, requestData);
        },
        [this, filePath](PersisterRequestId const& requestId) {
            _PersisterFacade::get()->fetchSaveSimulationData(requestId);
            _context->showMessage("Simulation saved");
            return McpToolResult{.text = std::format("Saved the simulation to '{}'.", filePath)};
        },
        completion);
    _context->showMessage("Saving ...");
}

void McpSimulationTools::loadSimulation(boost::json::object const& arguments, McpToolCompletion const& completion)
{
    auto filePath = McpArguments::getString(arguments, "file_path");
    auto path = McpArguments::getFilePath(arguments, "file_path");
    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument(std::format("The file '{}' does not exist.", filePath));
    }

    _loadTask.execute(
        [path](SenderId const& senderId) {
            return _PersisterFacade::get()->scheduleReadSimulation(
                SenderInfo{.senderId = senderId, .wishResultData = true, .wishErrorInfo = true}, ReadSimulationRequestData{.filename = path});
        },
        [this, filePath](PersisterRequestId const& requestId) {
            auto data = _PersisterFacade::get()->fetchReadSimulationData(requestId);
            _context->applySimulation(data.simulationDesc);
            _context->onSelectionChanged();
            _context->showMessage(data.filename.string());
            auto const& worldSize = data.simulationDesc._worldSize;
            return McpToolResult{
                .text = std::format(
                    "Loaded the simulation '{}' with a world size of {} x {} at time step {}. The simulation is paused.",
                    filePath,
                    worldSize.x,
                    worldSize.y,
                    data.simulationDesc._timestep)};
        },
        completion);
    _context->showMessage("Loading ...");
}
