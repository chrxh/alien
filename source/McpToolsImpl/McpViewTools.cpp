#include "McpViewTools.h"

#include <algorithm>
#include <cmath>
#include <ranges>
#include <format>

#include <boost/json.hpp>

#include <Base/ExitScopeGuard.h>
#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/StatisticsEntry.h>

#include <Network/McpArguments.h>
#include <Network/McpJson.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr MaxPictureSize = 2048;
    auto constexpr DefaultPictureWidth = 1024;
    auto constexpr DefaultMaxLineages = 10;
    auto constexpr MaxLineages = 100;
    auto constexpr JpgMimeType = "image/jpeg";
    auto constexpr PngMimeType = "image/png";

    boost::json::object withView(boost::json::object properties)
    {
        properties["center_x"] = McpSchema::number("X coordinate of the center of the view, default: current center");
        properties["center_y"] = McpSchema::number("Y coordinate of the center of the view, default: current center");
        properties["visible_width"] = McpSchema::number("Width of the visible world area, default: current width", 1.0);
        return properties;
    }
}

std::vector<McpTool> McpViewTools::getTools(McpToolContext& context)
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
            .name = "get_statistics",
            .description = "Returns the numbers of objects by type, the numbers of energy particles, the total energy and the largest lineages (groups "
                           "of related creatures) with their sizes. The representative cell of a lineage belongs to its most advanced creature and "
                           "can be inspected with inspect_objects.",
            .inputSchema = McpSchema::object(
                {{"max_lineages", McpSchema::integer(std::format("Maximum number of lineages to return, default: {}", DefaultMaxLineages), 0, MaxLineages)}}),
            .handler = [this](boost::json::object const& arguments) { return getStatistics(arguments); },
        },
        McpTool{
            .name = "set_view",
            .description = "Moves the view of ALIEN to another world position and/or changes the zoom. The zoom is given as the width of the visible "
                           "world area.",
            .inputSchema = McpSchema::object(withView({})),
            .handler = [this](boost::json::object const& arguments) { return setView(arguments); },
        },
        McpTool{
            .name = "take_screenshot",
            .description = "Renders the simulation as it is shown in ALIEN and returns the picture. Without view arguments the current view is "
                           "rendered. With view arguments the given area is rendered and the view of ALIEN is restored afterwards. The height of the "
                           "picture determines the visible height of the world area.",
            .inputSchema = McpSchema::object(withView({
                {"width", McpSchema::integer(std::format("Picture width in pixels, default: {}", DefaultPictureWidth), 16, MaxPictureSize)},
                {"height", McpSchema::integer("Picture height in pixels, default: according to the aspect ratio of the view", 16, MaxPictureSize)},
                {"format", McpSchema::enumeration("Picture format, default: jpeg", {"jpeg", "png"})},
            })),
            .handler = [this](boost::json::object const& arguments) { return takeScreenshot(arguments); },
        },
    };
}

McpToolResult McpViewTools::getSimulationInfo() const
{
    auto worldSize = _SimulationFacade::get()->getWorldSize();
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
        {"visible_area", describeVisibleArea()},
        {"selection", boost::json::object{{"objects", selection.numObjects}, {"energy_particles", selection.numEnergyParticles}}},
        {"colors_by_index", std::move(colors)},
    };
    return {.text = McpJson::serialize(result)};
}

McpToolResult McpViewTools::getStatistics(boost::json::object const& arguments) const
{
    auto maxLineages = McpArguments::getOptionalInt(arguments, "max_lineages", 0, MaxLineages).value_or(DefaultMaxLineages);
    auto statistics = _SimulationFacade::get()->getStatisticsEntry();
    auto const& objects = statistics.objectStatistics;

    auto lineages = statistics.lineageEntries;
    std::ranges::sort(lineages, std::ranges::greater(), &LineageStatisticsEntry::numCreatures);
    boost::json::array lineageArray;
    for (auto const& lineage : lineages | std::views::take(maxLineages)) {
        auto average = [](auto sum, auto count) { return count > 0 ? toDouble(sum) / count : 0.0; };
        boost::json::object entry{
            {"lineage_id", lineage.lineageId},
            {"creatures", lineage.numCreatures},
            {"genomes", lineage.numGenomes},
            {"average_cells_per_creature", average(lineage.sumCreatureCells, lineage.numCreatures)},
            {"average_generation", average(lineage.sumCreatureGenerations, lineage.numCreatures)},
            {"average_genome_nodes", average(lineage.sumGenomeNodes, lineage.numGenomes)},
            {"created_creatures", lineage.numCreatedCreatures},
        };
        if (lineage.representativeCellId != 0) {
            entry["representative_cell_id"] = std::to_string(lineage.representativeCellId);
        }
        lineageArray.emplace_back(std::move(entry));
    }

    auto result = boost::json::object{
        {"time_step", _SimulationFacade::get()->getCurrentTimestep()},
        {"solid_objects", objects.numSolidObjects},
        {"fluid_objects", objects.numFluidObjects},
        {"free_cells", objects.numFreeCellObjects},
        {"cells", objects.numCellObjects},
        {"energy_particles", objects.numEnergyParticles},
        {"total_energy", objects.totalInternalEnergy},
        {"lineages", lineages.size()},
        {"largest_lineages", std::move(lineageArray)},
    };
    return {.text = McpJson::serialize(result)};
}

McpToolResult McpViewTools::setView(boost::json::object const& arguments) const
{
    applyView(arguments);
    return {.text = McpJson::serialize(boost::json::object{{"visible_area", describeVisibleArea()}})};
}

McpToolResult McpViewTools::takeScreenshot(boost::json::object const& arguments) const
{
    auto origCenter = _context->getVisibleAreaCenter();
    auto origZoomFactor = _context->getZoomFactor();
    ExitScopeGuard restoreView([&] { _context->setVisibleArea(origCenter, origZoomFactor); });
    applyView(arguments);

    auto visibleAreaSize = _context->getVisibleAreaSize();
    auto width = McpArguments::getOptionalInt(arguments, "width", 16, MaxPictureSize).value_or(DefaultPictureWidth);
    auto defaultHeight = std::clamp(toInt(std::round(toFloat(width) * visibleAreaSize.y / visibleAreaSize.x)), 16, MaxPictureSize);
    auto height = McpArguments::getOptionalInt(arguments, "height", 16, MaxPictureSize).value_or(defaultHeight);
    auto png = McpArguments::getOptionalString(arguments, "format").value_or("jpeg") == "png";

    auto picture = _context->createPicture({width, height}, png ? McpPictureFormat::Png : McpPictureFormat::Jpg);

    auto center = _context->getVisibleAreaCenter();
    auto visibleHeight = visibleAreaSize.x * toFloat(height) / toFloat(width);
    return {
        .text = std::format(
            "Screenshot of {} x {} pixels showing the world area with center ({:.1f}, {:.1f}), width {:.1f} and height {:.1f}.",
            width,
            height,
            center.x,
            center.y,
            visibleAreaSize.x,
            visibleHeight),
        .images = {McpImage{.mimeType = png ? PngMimeType : JpgMimeType, .data = std::move(picture)}},
    };
}

void McpViewTools::applyView(boost::json::object const& arguments) const
{
    auto center = _context->getVisibleAreaCenter();
    center.x = McpArguments::getOptionalFloat(arguments, "center_x").value_or(center.x);
    center.y = McpArguments::getOptionalFloat(arguments, "center_y").value_or(center.y);

    auto zoomFactor = _context->getZoomFactor();
    if (auto visibleWidth = McpArguments::getOptionalFloat(arguments, "visible_width", 1.0f)) {
        zoomFactor = zoomFactor * _context->getVisibleAreaSize().x / *visibleWidth;
    }
    _context->setVisibleArea(center, zoomFactor);
}

boost::json::object McpViewTools::describeVisibleArea() const
{
    auto center = _context->getVisibleAreaCenter();
    auto size = _context->getVisibleAreaSize();
    return {
        {"center_x", center.x},
        {"center_y", center.y},
        {"width", size.x},
        {"height", size.y},
    };
}
