#include "McpSelectionTools.h"

#include <algorithm>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/StringHelper.h>

#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/ShallowUpdateSelectionData.h>
#include <EngineInterface/SimulationFacade.h>

#include <Network/McpArguments.h>
#include <Network/McpSchema.h>

namespace
{
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

std::vector<McpTool> McpSelectionTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "select_area",
            .description = "Selects all objects and energy particles inside a rectangular area. The previous selection is replaced. The editing tools "
                           "(delete_selection, fix_selection, color_selection, ...) operate on the selection.",
            .inputSchema = McpSchema::object(
                {
                    {"x1", McpSchema::number("X coordinate of one corner")},
                    {"y1", McpSchema::number("Y coordinate of one corner")},
                    {"x2", McpSchema::number("X coordinate of the opposite corner")},
                    {"y2", McpSchema::number("Y coordinate of the opposite corner")},
                },
                {"x1", "y1", "x2", "y2"}),
            .handler = [this](boost::json::object const& arguments) { return selectArea(arguments); },
        },
        McpTool{
            .name = "clear_selection",
            .description = "Deselects everything.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return clearSelection(); },
        },
        McpTool{
            .name = "get_selection",
            .description = "Returns the size, center and velocity of the current selection.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return getSelection(); },
        },
        McpTool{
            .name = "delete_selection",
            .description = "Deletes the selected objects and energy particles.",
            .inputSchema = McpSchema::object(withIncludeClusters({})),
            .handler = [this](boost::json::object const& arguments) { return deleteSelection(arguments); },
        },
        McpTool{
            .name = "fix_selection",
            .description = "Fixes the selected objects in place or releases them again.",
            .inputSchema = McpSchema::object(withIncludeClusters({{"fixed", McpSchema::boolean("true to fix, false to release")}}), {"fixed"}),
            .handler = [this](boost::json::object const& arguments) { return fixSelection(arguments); },
        },
        McpTool{
            .name = "color_selection",
            .description = "Changes the color of the selected objects.",
            .inputSchema =
                McpSchema::object(withIncludeClusters({{"color", McpSchema::integer("Color index, see get_simulation_info", 0, MAX_COLORS - 1)}}), {"color"}),
            .handler = [this](boost::json::object const& arguments) { return colorSelection(arguments); },
        },
        McpTool{
            .name = "set_selection_sticky",
            .description = "Makes the selected objects sticky or unsticky. Sticky objects attach to other objects on contact.",
            .inputSchema = McpSchema::object(withIncludeClusters({{"sticky", McpSchema::boolean("true for sticky, false for unsticky")}}), {"sticky"}),
            .handler = [this](boost::json::object const& arguments) { return setSelectionSticky(arguments); },
        },
        McpTool{
            .name = "move_selection",
            .description = "Moves the selected objects by an offset. Without include_clusters, connections to unselected objects are cut.",
            .inputSchema = McpSchema::object(
                withIncludeClusters({{"dx", McpSchema::number("Offset in x direction")}, {"dy", McpSchema::number("Offset in y direction")}}), {"dx", "dy"}),
            .handler = [this](boost::json::object const& arguments) { return moveSelection(arguments); },
        },
        McpTool{
            .name = "rotate_selection",
            .description = "Rotates the selected objects around their center. Without include_clusters, connections to unselected objects are cut.",
            .inputSchema = McpSchema::object(withIncludeClusters({{"angle", McpSchema::number("Rotation angle in degrees", -360, 360)}}), {"angle"}),
            .handler = [this](boost::json::object const& arguments) { return rotateSelection(arguments); },
        },
        McpTool{
            .name = "relax_selection",
            .description = "Releases the mechanical stresses of the selected objects by adopting the current distances as the rest distances of their "
                           "connections.",
            .inputSchema = McpSchema::object(withIncludeClusters({})),
            .handler = [this](boost::json::object const& arguments) { return relaxSelection(arguments); },
        },
    };
}

McpToolResult McpSelectionTools::selectArea(boost::json::object const& arguments) const
{
    auto x1 = McpArguments::getFloat(arguments, "x1");
    auto y1 = McpArguments::getFloat(arguments, "y1");
    auto x2 = McpArguments::getFloat(arguments, "x2");
    auto y2 = McpArguments::getFloat(arguments, "y2");
    _SimulationFacade::get()->setSelection({std::min(x1, x2), std::min(y1, y2)}, {std::max(x1, x2), std::max(y1, y2)});
    _context->onSelectionChanged();
    return {.text = describeSelection(_SimulationFacade::get()->getSelectionShallowData())};
}

McpToolResult McpSelectionTools::clearSelection() const
{
    _SimulationFacade::get()->removeSelection();
    _context->onSelectionChanged();
    return {.text = "The selection is empty."};
}

McpToolResult McpSelectionTools::getSelection() const
{
    auto selection = _SimulationFacade::get()->getSelectionShallowData();
    auto result = boost::json::object{
        {"objects", selection.numObjects},
        {"objects_including_clusters", selection.numClusterCells},
        {"creatures", selection.numCreatures},
        {"energy_particles", selection.numEnergyParticles},
        {"center", boost::json::array{selection.centerPosX, selection.centerPosY}},
        {"velocity", boost::json::array{selection.centerVelX, selection.centerVelY}},
    };
    return {.text = boost::json::serialize(result)};
}

McpToolResult McpSelectionTools::deleteSelection(boost::json::object const& arguments) const
{
    auto includeClusters = getIncludeClusters(arguments);
    auto selection = getNonEmptySelection();
    _SimulationFacade::get()->removeSelectedObjects(includeClusters);
    _context->onSelectionChanged();
    return {
        .text = std::format(
            "Deleted {} objects and {} energy particles.",
            StringHelper::format(static_cast<uint64_t>(includeClusters ? selection.numClusterCells : selection.numObjects)),
            StringHelper::format(static_cast<uint64_t>(selection.numEnergyParticles)))};
}

McpToolResult McpSelectionTools::fixSelection(boost::json::object const& arguments) const
{
    auto fixed = McpArguments::getBool(arguments, "fixed");
    auto includeClusters = getIncludeClusters(arguments);
    getNonEmptySelection();
    _SimulationFacade::get()->setStatic(fixed, includeClusters);
    _context->onSelectionChanged();
    return {.text = fixed ? "The selected objects are fixed." : "The selected objects are released."};
}

McpToolResult McpSelectionTools::colorSelection(boost::json::object const& arguments) const
{
    auto color = McpArguments::getInt(arguments, "color", 0, MAX_COLORS - 1);
    auto includeClusters = getIncludeClusters(arguments);
    getNonEmptySelection();
    _SimulationFacade::get()->colorSelectedObjects(static_cast<unsigned char>(color), includeClusters);
    _context->onSelectionChanged();
    return {.text = std::format("The selected objects have color {}.", color)};
}

McpToolResult McpSelectionTools::setSelectionSticky(boost::json::object const& arguments) const
{
    auto sticky = McpArguments::getBool(arguments, "sticky");
    auto includeClusters = getIncludeClusters(arguments);
    getNonEmptySelection();
    if (sticky) {
        _SimulationFacade::get()->makeSticky(includeClusters);
    } else {
        _SimulationFacade::get()->removeStickiness(includeClusters);
    }
    _context->onSelectionChanged();
    return {.text = sticky ? "The selected objects are sticky." : "The selected objects are unsticky."};
}

McpToolResult McpSelectionTools::moveSelection(boost::json::object const& arguments) const
{
    auto dx = McpArguments::getFloat(arguments, "dx");
    auto dy = McpArguments::getFloat(arguments, "dy");
    auto includeClusters = getIncludeClusters(arguments);
    auto selection = getNonEmptySelection();

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = includeClusters;
    updateData.posDeltaX = dx;
    updateData.posDeltaY = dy;
    updateData.velX = includeClusters ? selection.clusterCenterVelX : selection.centerVelX;
    updateData.velY = includeClusters ? selection.clusterCenterVelY : selection.centerVelY;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    _context->onSelectionChanged();
    return {.text = std::format("Moved the selection by ({}, {}).", StringHelper::format(dx, 1), StringHelper::format(dy, 1))};
}

McpToolResult McpSelectionTools::rotateSelection(boost::json::object const& arguments) const
{
    auto angle = McpArguments::getFloat(arguments, "angle", -360.0f, 360.0f);
    auto includeClusters = getIncludeClusters(arguments);
    getNonEmptySelection();

    ShallowUpdateSelectionData updateData;
    updateData.considerClusters = includeClusters;
    updateData.angleDelta = angle;
    _SimulationFacade::get()->shallowUpdateSelectedObjects(updateData);
    _context->onSelectionChanged();
    return {.text = std::format("Rotated the selection by {} degrees.", StringHelper::format(angle, 1))};
}

McpToolResult McpSelectionTools::relaxSelection(boost::json::object const& arguments) const
{
    auto includeClusters = getIncludeClusters(arguments);
    getNonEmptySelection();
    _SimulationFacade::get()->relaxSelectedObjects(includeClusters);
    _context->onSelectionChanged();
    return {.text = "Released the stresses of the selected objects."};
}

SelectionShallowData McpSelectionTools::getNonEmptySelection() const
{
    auto result = _SimulationFacade::get()->getSelectionShallowData();
    if (result.numObjects == 0 && result.numEnergyParticles == 0) {
        throw std::invalid_argument("Nothing is selected. Select an area with select_area first.");
    }
    return result;
}

std::string McpSelectionTools::describeSelection(SelectionShallowData const& selection) const
{
    auto result = std::format(
        "Selected {} objects and {} energy particles.",
        StringHelper::format(static_cast<uint64_t>(selection.numObjects)),
        StringHelper::format(static_cast<uint64_t>(selection.numEnergyParticles)));
    if (selection.numClusterCells > selection.numObjects) {
        result += std::format(
            " With include_clusters, {} objects of connected cell networks are affected.",
            StringHelper::format(static_cast<uint64_t>(selection.numClusterCells)));
    }
    return result;
}
