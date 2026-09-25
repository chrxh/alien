#include "McpCreatorTools.h"

#include <cmath>
#include <filesystem>
#include <ranges>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/Math.h>
#include <Base/StringHelper.h>

#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/SimulationFacade.h>

#include <Network/McpArguments.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr MaxObjectsPerCommand = 1000000.0f;
    auto constexpr AreaDensityFactor = 1.2f;
    auto constexpr DefaultObjectDistance = 1.0f;
    auto constexpr MinObjectDistance = 0.5f;
    auto constexpr MaxObjectDistance = 10.0f;
    auto constexpr MinPencilRadius = 1.0f;
    auto constexpr MaxPencilRadius = 8.0f;

    auto const MaterialNames = std::vector<std::string>{"solid", "fluid", "free_cell", "energy_particle"};

    boost::json::object withObjectProperties(boost::json::object properties)
    {
        properties["material"] = McpSchema::enumeration(
            "solid: elastic soft body material, fluid: liquid-like material, free_cell: cells that do not belong to creatures, energy_particle: free "
            "energy. Default: solid",
            MaterialNames);
        properties["color"] = McpSchema::integer("Color index, see get_simulation_info. Default: 0", 0, MAX_COLORS - 1);
        properties["energy"] = McpSchema::number("Energy per object. Default: 100", 0);
        properties["stiffness"] = McpSchema::number("Rigidity of solid objects and free cells. Default: 1", 0, 1);
        properties["glow"] = McpSchema::number("Glow of fluid objects. Default: 0", 0, 1);
        properties["is_static"] = McpSchema::boolean("Static objects do not move. Default: false");
        properties["sticky"] = McpSchema::boolean("Sticky objects attach to other objects on contact. Default: false");
        return properties;
    }

    boost::json::object withCenter(boost::json::object properties)
    {
        properties["center_x"] = McpSchema::number("X coordinate of the center. Default: center of the visible area");
        properties["center_y"] = McpSchema::number("Y coordinate of the center. Default: center of the visible area");
        return properties;
    }

    boost::json::object withObjectDistance(boost::json::object properties)
    {
        properties["object_distance"] = McpSchema::number("Distance between neighboring objects. Default: 1", MinObjectDistance, MaxObjectDistance);
        return properties;
    }

    float getObjectDistance(boost::json::object const& arguments)
    {
        return McpArguments::getOptionalFloat(arguments, "object_distance", MinObjectDistance, MaxObjectDistance).value_or(DefaultObjectDistance);
    }

    float calcLength(std::vector<RealVector2D> const& path)
    {
        auto result = 0.0f;
        for (auto const& [from, to] : std::views::zip(path, path | std::views::drop(1))) {
            result += Math::length(to - from);
        }
        return result;
    }

    std::string formatPos(RealVector2D const& pos)
    {
        return std::format("({}, {})", StringHelper::format(pos.x, 1), StringHelper::format(pos.y, 1));
    }
}

std::vector<McpTool> McpCreatorTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "create_object",
            .description = "Creates a single object or energy particle.",
            .inputSchema = McpSchema::object(withObjectProperties({
                {"x", McpSchema::number("X coordinate. Default: center of the visible area")},
                {"y", McpSchema::number("Y coordinate. Default: center of the visible area")},
            })),
            .handler = [this](boost::json::object const& arguments) { return createObject(arguments); },
        },
        McpTool{
            .name = "create_rectangle",
            .description = "Creates a rectangular network of connected objects.",
            .inputSchema = McpSchema::object(
                withObjectProperties(withObjectDistance(withCenter({
                    {"horizontal_objects", McpSchema::integer("Number of objects in x direction", 1)},
                    {"vertical_objects", McpSchema::integer("Number of objects in y direction", 1)},
                }))),
                {"horizontal_objects", "vertical_objects"}),
            .handler = [this](boost::json::object const& arguments) { return createRectangle(arguments); },
        },
        McpTool{
            .name = "create_hexagon",
            .description = "Creates a hexagonal network of connected objects.",
            .inputSchema = McpSchema::object(
                withObjectProperties(withObjectDistance(withCenter({{"layers", McpSchema::integer("Number of layers around the center", 1)}}))), {"layers"}),
            .handler = [this](boost::json::object const& arguments) { return createHexagon(arguments); },
        },
        McpTool{
            .name = "create_disc",
            .description = "Creates a disc-shaped network of connected objects. With an inner radius greater than 0, a ring is created.",
            .inputSchema = McpSchema::object(
                withObjectProperties(withObjectDistance(withCenter({
                    {"outer_radius", McpSchema::number("Outer radius", 0)},
                    {"inner_radius", McpSchema::number("Inner radius. Default: 0", 0)},
                }))),
                {"outer_radius"}),
            .handler = [this](boost::json::object const& arguments) { return createDisc(arguments); },
        },
        McpTool{
            .name = "draw_freehand",
            .description = "Draws a stroke with a round pencil through the given points, like the freehand drawing in the creator window.",
            .inputSchema = McpSchema::object(
                withObjectProperties({
                    {"points", McpSchema::points("Points of the stroke as [x, y] pairs", 1)},
                    {"pencil_radius", McpSchema::number("Radius of the pencil. Default: 1", MinPencilRadius, MaxPencilRadius)},
                }),
                {"points"}),
            .handler = [this](boost::json::object const& arguments) { return drawFreehand(arguments); },
        },
        McpTool{
            .name = "create_line",
            .description = "Creates a network of connected objects along a polyline.",
            .inputSchema = McpSchema::object(
                withObjectProperties(withObjectDistance({{"points", McpSchema::points("Points of the polyline as [x, y] pairs", 2)}})), {"points"}),
            .handler = [this](boost::json::object const& arguments) { return createLine(arguments); },
        },
        McpTool{
            .name = "create_curve",
            .description = "Creates a network of connected objects along a Bezier curve.",
            .inputSchema = McpSchema::object(
                withObjectProperties(withObjectDistance({{"control_points", McpSchema::points("Control points of the Bezier curve as [x, y] pairs", 2)}})),
                {"control_points"}),
            .handler = [this](boost::json::object const& arguments) { return createCurve(arguments); },
        },
        McpTool{
            .name = "create_polygon",
            .description = "Fills a polygon with a network of connected objects.",
            .inputSchema = McpSchema::object(
                withObjectProperties(withObjectDistance({{"points", McpSchema::points("Corners of the polygon as [x, y] pairs", 3)}})), {"points"}),
            .handler = [this](boost::json::object const& arguments) { return createPolygon(arguments); },
        },
        McpTool{
            .name = "create_pattern_from_image",
            .description = "Converts an image file (e.g. PNG) on this computer into a network of solid objects. Each bright pixel becomes an object whose "
                           "color is the closest color of the palette.",
            .inputSchema = McpSchema::object(withCenter({{"file_path", McpSchema::string("Absolute path of the image file")}}), {"file_path"}),
            .handler = [this](boost::json::object const& arguments) { return createPatternFromImage(arguments); },
        },
    };
}

McpToolResult McpCreatorTools::createObject(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto pos = _context->getVisibleAreaCenter();
    auto x = McpArguments::getOptionalFloat(arguments, "x");
    auto y = McpArguments::getOptionalFloat(arguments, "y");
    if (x.has_value() != y.has_value()) {
        throw std::invalid_argument("Specify both 'x' and 'y' or neither.");
    }
    if (x) {
        pos = {*x, *y};
    }
    checkInsideWorld(pos);
    return addToSimulation(CreatorService::get().createSingleObject(properties, pos), properties, "a single object at " + formatPos(pos));
}

McpToolResult McpCreatorTools::createRectangle(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto center = getCenter(arguments);
    auto numObjects = IntVector2D{McpArguments::getInt(arguments, "horizontal_objects", 1), McpArguments::getInt(arguments, "vertical_objects", 1)};
    checkNumObjects(toFloat(numObjects.x) * toFloat(numObjects.y));
    return addToSimulation(
        CreatorService::get().createRectangle(properties, center, numObjects, getObjectDistance(arguments)),
        properties,
        "a rectangle centered at " + formatPos(center));
}

McpToolResult McpCreatorTools::createHexagon(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto center = getCenter(arguments);
    auto layers = McpArguments::getInt(arguments, "layers", 1);
    checkNumObjects(3.0f * toFloat(layers) * toFloat(layers));
    return addToSimulation(
        CreatorService::get().createHexagon(properties, center, layers, getObjectDistance(arguments)),
        properties,
        "a hexagon centered at " + formatPos(center));
}

McpToolResult McpCreatorTools::createDisc(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto center = getCenter(arguments);
    auto outerRadius = McpArguments::getFloat(arguments, "outer_radius", 0);
    auto innerRadius = McpArguments::getOptionalFloat(arguments, "inner_radius", 0).value_or(0.0f);
    if (innerRadius > outerRadius) {
        throw std::invalid_argument("'inner_radius' must not be greater than 'outer_radius'.");
    }
    auto objectDistance = getObjectDistance(arguments);
    checkNumObjects(Const::Pi * (outerRadius * outerRadius - innerRadius * innerRadius) / (objectDistance * objectDistance) * AreaDensityFactor);
    return addToSimulation(
        CreatorService::get().createDisc(properties, center, outerRadius, innerRadius, objectDistance),
        properties,
        std::string(innerRadius > 0 ? "a ring" : "a disc") + " centered at " + formatPos(center));
}

McpToolResult McpCreatorTools::drawFreehand(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto points = getPointsInsideWorld(arguments, "points", 1);
    auto pencilRadius = McpArguments::getOptionalFloat(arguments, "pencil_radius", MinPencilRadius, MaxPencilRadius).value_or(MinPencilRadius);
    auto pencilDiameter = 2 * pencilRadius + 1;
    checkNumObjects((calcLength(points) + 1) * pencilDiameter * pencilDiameter);
    return addToSimulation(CreatorService::get().createFreehandStroke(properties, points, pencilRadius), properties, "a freehand stroke");
}

McpToolResult McpCreatorTools::createLine(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto points = getPointsInsideWorld(arguments, "points", 2);
    auto objectDistance = getObjectDistance(arguments);
    checkNumObjects(calcLength(points) / objectDistance);
    return addToSimulation(CreatorService::get().createLine(properties, points, objectDistance), properties, "a line");
}

McpToolResult McpCreatorTools::createCurve(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto controlPoints = getPointsInsideWorld(arguments, "control_points", 2);
    auto objectDistance = getObjectDistance(arguments);
    checkNumObjects(calcLength(controlPoints) / objectDistance);
    return addToSimulation(CreatorService::get().createCurve(properties, controlPoints, objectDistance), properties, "a curve");
}

McpToolResult McpCreatorTools::createPolygon(boost::json::object const& arguments) const
{
    auto properties = getObjectProperties(arguments);
    auto points = getPointsInsideWorld(arguments, "points", 3);
    auto objectDistance = getObjectDistance(arguments);

    auto area = 0.0f;
    auto closedPolygon = points;
    closedPolygon.emplace_back(points.front());
    for (auto const& [from, to] : std::views::zip(closedPolygon, closedPolygon | std::views::drop(1))) {
        area += from.x * to.y - to.x * from.y;
    }
    checkNumObjects(std::abs(area) / 2 / (objectDistance * objectDistance) * AreaDensityFactor);
    return addToSimulation(CreatorService::get().createPolygon(properties, points, objectDistance), properties, "a polygon");
}

McpToolResult McpCreatorTools::createPatternFromImage(boost::json::object const& arguments) const
{
    auto filePath = McpArguments::getString(arguments, "file_path");
    auto path = std::filesystem::path(std::u8string(filePath.begin(), filePath.end()));
    if (!std::filesystem::is_regular_file(path)) {
        throw std::invalid_argument(std::format("The file '{}' does not exist.", filePath));
    }
    auto center = getCenter(arguments);
    auto image = _context->loadImage(path);
    if (!image) {
        throw std::invalid_argument(std::format("The file '{}' could not be read as an image.", filePath));
    }
    return addToSimulation(
        CreatorService::get().createPatternFromImage(*image, center),
        CreatorService::ObjectProperties(),
        "a pattern from the image centered at " + formatPos(center));
}

CreatorService::ObjectProperties McpCreatorTools::getObjectProperties(boost::json::object const& arguments) const
{
    auto materialName = McpArguments::getOptionalString(arguments, "material").value_or("solid");
    auto material = std::ranges::find(MaterialNames, materialName);
    if (material == MaterialNames.end()) {
        throw std::invalid_argument("'material' must be one of solid, fluid, free_cell, energy_particle.");
    }
    return CreatorService::ObjectProperties()
        .material(toInt(std::distance(MaterialNames.begin(), material)))
        .color(McpArguments::getOptionalInt(arguments, "color", 0, MAX_COLORS - 1).value_or(0))
        .energy(McpArguments::getOptionalFloat(arguments, "energy", 0).value_or(100.0f))
        .stiffness(McpArguments::getOptionalFloat(arguments, "stiffness", 0, 1).value_or(1.0f))
        .glow(McpArguments::getOptionalFloat(arguments, "glow", 0, 1).value_or(0.0f))
        .isStatic(McpArguments::getOptionalBool(arguments, "is_static").value_or(false))
        .sticky(McpArguments::getOptionalBool(arguments, "sticky").value_or(false));
}

RealVector2D McpCreatorTools::getCenter(boost::json::object const& arguments) const
{
    auto x = McpArguments::getOptionalFloat(arguments, "center_x");
    auto y = McpArguments::getOptionalFloat(arguments, "center_y");
    if (x.has_value() != y.has_value()) {
        throw std::invalid_argument("Specify both 'center_x' and 'center_y' or neither.");
    }
    auto result = x ? RealVector2D{*x, *y} : _context->getVisibleAreaCenter();
    checkInsideWorld(result);
    return result;
}

std::vector<RealVector2D> McpCreatorTools::getPointsInsideWorld(boost::json::object const& arguments, std::string const& key, size_t minNumPoints) const
{
    auto result = McpArguments::getPoints(arguments, key, minNumPoints);
    for (auto const& point : result) {
        checkInsideWorld(point);
    }
    return result;
}

void McpCreatorTools::checkInsideWorld(RealVector2D const& pos) const
{
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    if (pos.x < 0 || pos.y < 0 || pos.x >= toFloat(worldSize.x) || pos.y >= toFloat(worldSize.y)) {
        throw std::invalid_argument(std::format("The position {} lies outside the world of size {} x {}.", formatPos(pos), worldSize.x, worldSize.y));
    }
}

void McpCreatorTools::checkNumObjects(float estimatedNumObjects) const
{
    if (estimatedNumObjects > MaxObjectsPerCommand) {
        throw std::invalid_argument(std::format(
            "This would create about {} objects, but at most {} are allowed per command. Reduce the size or increase 'object_distance'.",
            StringHelper::format(static_cast<uint64_t>(estimatedNumObjects)),
            StringHelper::format(static_cast<uint64_t>(MaxObjectsPerCommand))));
    }
}

McpToolResult McpCreatorTools::addToSimulation(ContentDesc&& content, CreatorService::ObjectProperties const& properties, std::string const& shape) const
{
    auto numEntities = content._objects.size() + content._energies.size();
    if (numEntities == 0) {
        throw std::invalid_argument("No objects were created. Check the size and position parameters.");
    }
    checkNumObjects(toFloat(numEntities));

    _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
    _context->onSelectionChanged();

    auto entityName = [&] {
        switch (properties._material) {
        case CreationMaterial_Fluid:
            return "fluid object";
        case CreationMaterial_FreeCell:
            return "free cell";
        case CreationMaterial_EnergyParticle:
            return "energy particle";
        default:
            return "solid object";
        }
    }();
    return {
        .text = std::format(
            "Created {} with {} {}{}. The new objects are selected.", shape, StringHelper::format(numEntities), entityName, numEntities == 1 ? "" : "s")};
}
