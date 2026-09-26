#include "McpEditTools.h"

#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/Math.h>
#include <Base/StringHelper.h>

#include <Data/DescEditService.h>

#include <EngineInterface/ShallowUpdateSelectionData.h>
#include <EngineInterface/SimulationFacade.h>

#include <Network/McpArguments.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr DefaultSelectionRadius = 1.0f;
    auto constexpr MaxSelectionRadius = 100.0f;
    auto constexpr DefaultForceRadius = 5.0f;
    auto constexpr MaxForceRadius = 100.0f;
    auto constexpr DefaultForceStrength = 0.2f;
    auto constexpr MaxForceStrength = 5.0f;
    auto constexpr MaxVelocity = 10.0f;
    auto constexpr MaxAngularVelocity = 90.0f;

    boost::json::object withIncludeClusters(boost::json::object properties)
    {
        properties["include_clusters"] = McpSchema::boolean(
            "Also apply the change to all objects connected to the selected ones (cell networks), even if they lie outside the selected area. "
            "Default: false");
        return properties;
    }

    bool getIncludeClusters(boost::json::object const& arguments)
    {
        return McpArguments::getOptionalBool(arguments, "include_clusters").value_or(false);
    }
}

std::vector<McpTool> McpEditTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "select_at",
            .description = "Selects the object or energy particle closest to a position, like a click in ALIEN. By default the previous selection is "
                           "replaced. With toggle, the object is added to or removed from the selection.",
            .inputSchema = McpSchema::object(
                {
                    {"x", McpSchema::number("X coordinate")},
                    {"y", McpSchema::number("Y coordinate")},
                    {"radius",
                     McpSchema::number(
                         std::format("Maximum distance of the object to the position, default: {}", DefaultSelectionRadius), 0.1, MaxSelectionRadius)},
                    {"toggle", McpSchema::boolean("Add the object to the selection or remove it instead of replacing the selection. Default: false")},
                },
                {"x", "y"}),
            .handler = [this](boost::json::object const& arguments) { return selectAt(arguments); },
        },
        McpTool{
            .name = "glue_selection",
            .description = "Connects the selected objects that touch each other. Objects outside the selection are not connected.",
            .inputSchema = McpSchema::object(withIncludeClusters({})),
            .handler = [this](boost::json::object const& arguments) { return glueSelection(arguments); },
        },
        McpTool{
            .name = "connect_selection_to_surroundings",
            .description = "Connects the selected objects with all objects they touch, including unselected ones.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return connectSelectionToSurroundings(); },
        },
        McpTool{
            .name = "cut_connections",
            .description = "Cuts all connections crossing the line from (x1, y1) to (x2, y2), like the scissors in ALIEN.",
            .inputSchema = McpSchema::object(
                withIncludeClusters({
                    {"x1", McpSchema::number("X coordinate of the start point")},
                    {"y1", McpSchema::number("Y coordinate of the start point")},
                    {"x2", McpSchema::number("X coordinate of the end point")},
                    {"y2", McpSchema::number("Y coordinate of the end point")},
                    {"only_in_selection", McpSchema::boolean("Only cut connections of selected objects. Default: false")},
                }),
                {"x1", "y1", "x2", "y2"}),
            .handler = [this](boost::json::object const& arguments) { return cutConnections(arguments); },
        },
        McpTool{
            .name = "set_selection_velocity",
            .description = "Sets the velocity of the selected objects.",
            .inputSchema = McpSchema::object(
                withIncludeClusters({
                    {"vx", McpSchema::number("Velocity in x direction per time step", -MaxVelocity, MaxVelocity)},
                    {"vy", McpSchema::number("Velocity in y direction per time step", -MaxVelocity, MaxVelocity)},
                }),
                {"vx", "vy"}),
            .handler = [this](boost::json::object const& arguments) { return setSelectionVelocity(arguments); },
        },
        McpTool{
            .name = "set_selection_angular_velocity",
            .description = "Lets the selected objects rotate around their center.",
            .inputSchema = McpSchema::object(
                withIncludeClusters({{"angular_velocity", McpSchema::number("Rotation in degrees per time step", -MaxAngularVelocity, MaxAngularVelocity)}}),
                {"angular_velocity"}),
            .handler = [this](boost::json::object const& arguments) { return setSelectionAngularVelocity(arguments); },
        },
        McpTool{
            .name = "uniform_selection_velocities",
            .description = "Gives all selected objects their average velocity, so that they move together without rotating.",
            .inputSchema = McpSchema::object(withIncludeClusters({})),
            .handler = [this](boost::json::object const& arguments) { return uniformSelectionVelocities(arguments); },
        },
        McpTool{
            .name = "copy_selection",
            .description = "Copies the selected objects and energy particles to a clipboard of the MCP server, which paste_selection can insert.",
            .inputSchema = McpSchema::object(withIncludeClusters({})),
            .handler = [this](boost::json::object const& arguments) { return copySelection(arguments); },
        },
        McpTool{
            .name = "paste_selection",
            .description = "Inserts a copy of the content copied with copy_selection centered at a position and selects it. It can be pasted multiple "
                           "times.",
            .inputSchema = McpSchema::object({
                {"x", McpSchema::number("X coordinate of the center, default: center of the visible area")},
                {"y", McpSchema::number("Y coordinate of the center, default: center of the visible area")},
            }),
            .handler = [this](boost::json::object const& arguments) { return pasteSelection(arguments); },
        },
        McpTool{
            .name = "apply_force",
            .description = "Pushes the objects near the line from (x1, y1) to (x2, y2) in the direction of the line, like dragging with the force tool "
                           "in ALIEN. It takes effect in the next time step while the simulation is running.",
            .inputSchema = McpSchema::object(
                {
                    {"x1", McpSchema::number("X coordinate of the start point")},
                    {"y1", McpSchema::number("Y coordinate of the start point")},
                    {"x2", McpSchema::number("X coordinate of the end point")},
                    {"y2", McpSchema::number("Y coordinate of the end point")},
                    {"radius",
                     McpSchema::number(
                         std::format("Distance to the line within which objects are affected, default: {}", DefaultForceRadius), 0.1, MaxForceRadius)},
                    {"strength",
                     McpSchema::number(std::format("Velocity change of the affected objects, default: {}", DefaultForceStrength), 0.0, MaxForceStrength)},
                },
                {"x1", "y1", "x2", "y2"}),
            .handler = [this](boost::json::object const& arguments) { return applyForce(arguments); },
        },
    };
}

McpToolResult McpEditTools::selectAt(boost::json::object const& arguments) const
{
    RealVector2D pos{McpArguments::getFloat(arguments, "x"), McpArguments::getFloat(arguments, "y")};
    auto radius = McpArguments::getOptionalFloat(arguments, "radius", 0.1f, MaxSelectionRadius).value_or(DefaultSelectionRadius);
    if (McpArguments::getOptionalBool(arguments, "toggle").value_or(false)) {
        _SimulationFacade::get()->swapSelection(pos, radius);
    } else {
        _SimulationFacade::get()->switchSelection(pos, radius);
    }
    _context->onSelectionChanged();

    auto selection = _SimulationFacade::get()->getSelectionShallowData();
    return {
        .text = std::format(
            "The selection contains {} objects and {} energy particles.",
            StringHelper::format(static_cast<uint64_t>(selection.numObjects)),
            StringHelper::format(static_cast<uint64_t>(selection.numEnergyParticles)))};
}

McpToolResult McpEditTools::glueSelection(boost::json::object const& arguments) const
{
    auto includeClusters = getIncludeClusters(arguments);
    getNonEmptySelection();
    _SimulationFacade::get()->glueSelectedObjects(includeClusters);
    _context->onSelectionChanged();
    return {.text = "Connected the touching selected objects."};
}

McpToolResult McpEditTools::connectSelectionToSurroundings() const
{
    getNonEmptySelection();
    _SimulationFacade::get()->reconnectSelectedObjects();
    _context->onSelectionChanged();
    return {.text = "Connected the selected objects with the objects they touch."};
}

McpToolResult McpEditTools::cutConnections(boost::json::object const& arguments) const
{
    RealVector2D start{McpArguments::getFloat(arguments, "x1"), McpArguments::getFloat(arguments, "y1")};
    RealVector2D end{McpArguments::getFloat(arguments, "x2"), McpArguments::getFloat(arguments, "y2")};
    if (Math::length(end - start) < NEAR_ZERO) {
        throw std::invalid_argument("The start and end point of the cut must differ.");
    }
    auto onlyInSelection = McpArguments::getOptionalBool(arguments, "only_in_selection").value_or(false);
    _SimulationFacade::get()->cutConnections(start, end, onlyInSelection, getIncludeClusters(arguments));
    _context->onSelectionChanged();
    return {.text = std::format("Cut the connections crossing the line from ({}, {}) to ({}, {}).", start.x, start.y, end.x, end.y)};
}

McpToolResult McpEditTools::setSelectionVelocity(boost::json::object const& arguments) const
{
    auto vx = McpArguments::getFloat(arguments, "vx", -MaxVelocity, MaxVelocity);
    auto vy = McpArguments::getFloat(arguments, "vy", -MaxVelocity, MaxVelocity);
    getNonEmptySelection();

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = getIncludeClusters(arguments);
    updateData.velX = vx;
    updateData.velY = vy;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    _context->onSelectionChanged();
    return {.text = std::format("Set the velocity of the selection to ({}, {}).", vx, vy)};
}

McpToolResult McpEditTools::setSelectionAngularVelocity(boost::json::object const& arguments) const
{
    auto angularVelocity = McpArguments::getFloat(arguments, "angular_velocity", -MaxAngularVelocity, MaxAngularVelocity);
    getNonEmptySelection();

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = getIncludeClusters(arguments);
    updateData.angularVel = angularVelocity;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    _context->onSelectionChanged();
    return {.text = std::format("Set the angular velocity of the selection to {} degrees per time step.", angularVelocity)};
}

McpToolResult McpEditTools::uniformSelectionVelocities(boost::json::object const& arguments) const
{
    getNonEmptySelection();
    _SimulationFacade::get()->uniformVelocitiesForSelectedObjects(getIncludeClusters(arguments));
    _context->onSelectionChanged();
    return {.text = "The selected objects move with their average velocity."};
}

McpToolResult McpEditTools::copySelection(boost::json::object const& arguments)
{
    getNonEmptySelection();
    _copiedSelection = _SimulationFacade::get()->getSelectedSimulationData(getIncludeClusters(arguments));
    return {
        .text = std::format(
            "Copied {} objects and {} energy particles.",
            StringHelper::format(static_cast<uint64_t>(_copiedSelection->_objects.size())),
            StringHelper::format(static_cast<uint64_t>(_copiedSelection->_energies.size())))};
}

McpToolResult McpEditTools::pasteSelection(boost::json::object const& arguments) const
{
    if (!_copiedSelection) {
        throw std::runtime_error("Nothing has been copied. Call copy_selection first.");
    }
    auto center = _context->getVisibleAreaCenter();
    center.x = McpArguments::getOptionalFloat(arguments, "x").value_or(center.x);
    center.y = McpArguments::getOptionalFloat(arguments, "y").value_or(center.y);

    auto content = *_copiedSelection;
    DescEditService::get().setCenter(content, center);
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    _context->onSelectionChanged();
    _context->showMessage("Selection pasted");
    return {.text = std::format("Pasted the copied content at ({}, {}). It is selected now.", center.x, center.y)};
}

McpToolResult McpEditTools::applyForce(boost::json::object const& arguments) const
{
    RealVector2D start{McpArguments::getFloat(arguments, "x1"), McpArguments::getFloat(arguments, "y1")};
    RealVector2D end{McpArguments::getFloat(arguments, "x2"), McpArguments::getFloat(arguments, "y2")};
    auto length = Math::length(end - start);
    if (length < NEAR_ZERO) {
        throw std::invalid_argument("The start and end point must differ.");
    }
    auto radius = McpArguments::getOptionalFloat(arguments, "radius", 0.1f, MaxForceRadius).value_or(DefaultForceRadius);
    auto strength = McpArguments::getOptionalFloat(arguments, "strength", 0.0f, MaxForceStrength).value_or(DefaultForceStrength);
    _SimulationFacade::get()->applyForce_async(start, end, (end - start) / length * strength, radius);

    auto result = std::format("Applied a force along the line from ({}, {}) to ({}, {}).", start.x, start.y, end.x, end.y);
    if (!_SimulationFacade::get()->isSimulationRunning()) {
        result += " It takes effect as soon as the simulation runs.";
    }
    return {.text = result};
}

SelectionShallowData McpEditTools::getNonEmptySelection() const
{
    auto result = _SimulationFacade::get()->getSelectionShallowData();
    if (result.numObjects == 0 && result.numEnergyParticles == 0) {
        throw std::invalid_argument("Nothing is selected. Select objects with select_area or select_at first.");
    }
    return result;
}
