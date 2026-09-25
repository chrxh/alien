#include "McpSimulationTools.h"

#include <format>

#include <boost/json.hpp>

#include <Base/NameGeneratorService.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include <Network/McpArguments.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr MaxWorldSize = 10000;
}

std::vector<McpTool> McpSimulationTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "get_simulation_info",
            .description = "Returns the world size, whether the simulation is running, the time step, the area visible in ALIEN, the current selection "
                           "and the color palette. Call it before creating or selecting objects to know the coordinates.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return getSimulationInfo(); },
        },
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
            .name = "run_simulation",
            .description = "Starts the simulation in ALIEN.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return runSimulation(); },
        },
        McpTool{
            .name = "pause_simulation",
            .description = "Pauses the simulation in ALIEN.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return pauseSimulation(); },
        },
    };
}

McpToolResult McpSimulationTools::getSimulationInfo() const
{
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto visibleAreaCenter = _context->getVisibleAreaCenter();
    auto visibleAreaSize = _context->getVisibleAreaSize();
    auto selection = _SimulationFacade::get()->getSelectionShallowData();

    boost::json::array colors;
    for (auto const& color : _SimulationFacade::get()->getSimulationParameters().customizationColors.value.values) {
        colors.emplace_back(StringHelper::formatHexColor(color));
    }

    auto result = boost::json::object{
        {"world_width", worldSize.x},
        {"world_height", worldSize.y},
        {"running", _SimulationFacade::get()->isSimulationRunning()},
        {"time_step", _SimulationFacade::get()->getCurrentTimestep()},
        {"visible_area",
         boost::json::object{
             {"center_x", visibleAreaCenter.x},
             {"center_y", visibleAreaCenter.y},
             {"width", visibleAreaSize.x},
             {"height", visibleAreaSize.y},
         }},
        {"selection", boost::json::object{{"objects", selection.numObjects}, {"energy_particles", selection.numEnergyParticles}}},
        {"colors_by_index", std::move(colors)},
    };
    return {.text = boost::json::serialize(result)};
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

McpToolResult McpSimulationTools::runSimulation() const
{
    if (_SimulationFacade::get()->isSimulationRunning()) {
        return {.text = "The simulation is already running."};
    }
    _SimulationFacade::get()->runSimulation();
    _context->showMessage("Run");
    return {.text = "The simulation is running."};
}

McpToolResult McpSimulationTools::pauseSimulation() const
{
    if (!_SimulationFacade::get()->isSimulationRunning()) {
        return {.text = "The simulation is already paused."};
    }
    _SimulationFacade::get()->pauseSimulation();
    _context->showMessage("Pause");
    return {.text = "The simulation is paused."};
}
