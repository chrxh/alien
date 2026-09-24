#include "McpMultiplierTools.h"

#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/StringHelper.h>

#include <EngineInterface/SimulationFacade.h>

#include <Network/McpArguments.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr MaxObjectsPerCommand = 1000000.0f;

    void checkMinMax(float min, float max, std::string const& minKey, std::string const& maxKey)
    {
        if (min > max) {
            throw std::invalid_argument(std::format("'{}' must not be greater than '{}'.", minKey, maxKey));
        }
    }
}

std::vector<McpTool> McpMultiplierTools::getTools()
{
    auto const defaultGrid = DescEditService::GridMultiplyParameters();
    auto const defaultRandom = DescEditService::RandomMultiplyParameters();

    boost::json::object gridProperties;
    for (auto const& direction : {std::string("horizontal"), std::string("vertical")}) {
        auto axis = direction == "horizontal" ? "x" : "y";
        gridProperties[direction + "_copies"] = McpSchema::integer(std::format("Number of copies in {} direction, including the original", axis), 1);
        auto defaultDistance = direction == "horizontal" ? defaultGrid._horizontalDistance : defaultGrid._verticalDistance;
        gridProperties[direction + "_distance"] =
            McpSchema::number(std::format("Distance between neighboring copies in {} direction. Default: {}", axis, defaultDistance), 0);
        gridProperties[direction + "_angle_increment"] = McpSchema::number("Rotation added per copy in degrees. Default: 0");
        gridProperties[direction + "_velocity_x_increment"] = McpSchema::number("Velocity in x direction added per copy. Default: 0");
        gridProperties[direction + "_velocity_y_increment"] = McpSchema::number("Velocity in y direction added per copy. Default: 0");
        gridProperties[direction + "_angular_velocity_increment"] = McpSchema::number("Angular velocity added per copy. Default: 0");
    }

    return {
        McpTool{
            .name = "multiply_selection_grid",
            .description = "Arranges copies of the selection in a grid, like the grid multiplier in ALIEN. The selection always includes the connected "
                           "cell networks and is replaced by all copies, which are selected afterwards.",
            .inputSchema = McpSchema::object(gridProperties, {"horizontal_copies", "vertical_copies"}),
            .handler = [this](boost::json::object const& arguments) { return multiplyInGrid(arguments); },
        },
        McpTool{
            .name = "multiply_selection_random",
            .description = "Adds copies of the selection at random positions in the whole world, like the random multiplier in ALIEN. The selection "
                           "always includes the connected cell networks. The original and the copies are selected afterwards.",
            .inputSchema = McpSchema::object(
                {
                    {"copies", McpSchema::integer("Number of additional copies", 1)},
                    {"min_angle", McpSchema::number(std::format("Minimum rotation in degrees. Default: {}", defaultRandom._minAngle))},
                    {"max_angle", McpSchema::number(std::format("Maximum rotation in degrees. Default: {}", defaultRandom._maxAngle))},
                    {"min_velocity_x", McpSchema::number("Minimum velocity in x direction. Default: 0")},
                    {"max_velocity_x", McpSchema::number("Maximum velocity in x direction. Default: 0")},
                    {"min_velocity_y", McpSchema::number("Minimum velocity in y direction. Default: 0")},
                    {"max_velocity_y", McpSchema::number("Maximum velocity in y direction. Default: 0")},
                    {"min_angular_velocity", McpSchema::number("Minimum angular velocity. Default: 0")},
                    {"max_angular_velocity", McpSchema::number("Maximum angular velocity. Default: 0")},
                    {"overlapping_check", McpSchema::boolean("Avoid copies that overlap existing objects. Default: false")},
                },
                {"copies"}),
            .handler = [this](boost::json::object const& arguments) { return multiplyRandomly(arguments); },
        },
        McpTool{
            .name = "undo_multiplication",
            .description = "Reverts the last multiplication made via MCP, as long as the selection has not changed since then.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return undoMultiplication(); },
        },
    };
}

McpToolResult McpMultiplierTools::multiplyInGrid(boost::json::object const& arguments)
{
    auto defaults = DescEditService::GridMultiplyParameters();
    auto parameters = DescEditService::GridMultiplyParameters()
                          .horizontalNumber(McpArguments::getInt(arguments, "horizontal_copies", 1))
                          .horizontalDistance(McpArguments::getOptionalFloat(arguments, "horizontal_distance", 0).value_or(defaults._horizontalDistance))
                          .horizontalAngleInc(McpArguments::getOptionalFloat(arguments, "horizontal_angle_increment").value_or(0))
                          .horizontalVelXinc(McpArguments::getOptionalFloat(arguments, "horizontal_velocity_x_increment").value_or(0))
                          .horizontalVelYinc(McpArguments::getOptionalFloat(arguments, "horizontal_velocity_y_increment").value_or(0))
                          .horizontalAngularVelInc(McpArguments::getOptionalFloat(arguments, "horizontal_angular_velocity_increment").value_or(0))
                          .verticalNumber(McpArguments::getInt(arguments, "vertical_copies", 1))
                          .verticalDistance(McpArguments::getOptionalFloat(arguments, "vertical_distance", 0).value_or(defaults._verticalDistance))
                          .verticalAngleInc(McpArguments::getOptionalFloat(arguments, "vertical_angle_increment").value_or(0))
                          .verticalVelXinc(McpArguments::getOptionalFloat(arguments, "vertical_velocity_x_increment").value_or(0))
                          .verticalVelYinc(McpArguments::getOptionalFloat(arguments, "vertical_velocity_y_increment").value_or(0))
                          .verticalAngularVelInc(McpArguments::getOptionalFloat(arguments, "vertical_angular_velocity_increment").value_or(0));
    checkSelectionForMultiplication(parameters._horizontalNumber * parameters._verticalNumber);

    storeForUndo(MultiplierService::get().multiplyInGrid(parameters));
    return {.text = std::format("Arranged the selection in a {} x {} grid. {}", parameters._horizontalNumber, parameters._verticalNumber, describeSelection())};
}

McpToolResult McpMultiplierTools::multiplyRandomly(boost::json::object const& arguments)
{
    auto defaults = DescEditService::RandomMultiplyParameters();
    auto parameters = DescEditService::RandomMultiplyParameters()
                          .number(McpArguments::getInt(arguments, "copies", 1))
                          .minAngle(McpArguments::getOptionalFloat(arguments, "min_angle").value_or(defaults._minAngle))
                          .maxAngle(McpArguments::getOptionalFloat(arguments, "max_angle").value_or(defaults._maxAngle))
                          .minVelX(McpArguments::getOptionalFloat(arguments, "min_velocity_x").value_or(0))
                          .maxVelX(McpArguments::getOptionalFloat(arguments, "max_velocity_x").value_or(0))
                          .minVelY(McpArguments::getOptionalFloat(arguments, "min_velocity_y").value_or(0))
                          .maxVelY(McpArguments::getOptionalFloat(arguments, "max_velocity_y").value_or(0))
                          .minAngularVel(McpArguments::getOptionalFloat(arguments, "min_angular_velocity").value_or(0))
                          .maxAngularVel(McpArguments::getOptionalFloat(arguments, "max_angular_velocity").value_or(0))
                          .overlappingCheck(McpArguments::getOptionalBool(arguments, "overlapping_check").value_or(false));
    checkMinMax(parameters._minAngle, parameters._maxAngle, "min_angle", "max_angle");
    checkMinMax(parameters._minVelX, parameters._maxVelX, "min_velocity_x", "max_velocity_x");
    checkMinMax(parameters._minVelY, parameters._maxVelY, "min_velocity_y", "max_velocity_y");
    checkMinMax(parameters._minAngularVel, parameters._maxAngularVel, "min_angular_velocity", "max_angular_velocity");
    checkSelectionForMultiplication(parameters._number + 1);

    auto result = MultiplierService::get().multiplyRandomly(parameters);
    auto overlappingCheckSuccessful = result.overlappingCheckSuccessful;
    storeForUndo(std::move(result));

    auto text = std::format("Added {} copies at random positions. {}", parameters._number, describeSelection());
    if (!overlappingCheckSuccessful) {
        text += " Not all copies could be placed without overlapping existing objects.";
    }
    return {.text = text};
}

McpToolResult McpMultiplierTools::undoMultiplication()
{
    if (!_origSelection) {
        throw std::invalid_argument("There is no multiplication to undo.");
    }
    if (!_selectionAfterMultiplication->compareSizes(_SimulationFacade::get()->getSelectionShallowData())) {
        throw std::invalid_argument("The selection has changed since the last multiplication, so it can no longer be undone.");
    }
    MultiplierService::get().undo(*_origSelection);
    _origSelection.reset();
    _selectionAfterMultiplication.reset();
    return {.text = "Reverted the last multiplication. " + describeSelection()};
}

void McpMultiplierTools::checkSelectionForMultiplication(int numCopies) const
{
    auto selection = _SimulationFacade::get()->getSelectionShallowData();
    auto numEntities = selection.numClusterCells + selection.numEnergyParticles;
    if (numEntities == 0) {
        throw std::invalid_argument("Nothing is selected. Select an area with select_area first.");
    }
    auto estimatedNumEntities = toFloat(numEntities) * toFloat(numCopies);
    if (estimatedNumEntities > MaxObjectsPerCommand) {
        throw std::invalid_argument(std::format(
            "This would result in about {} objects, but at most {} are allowed per command.",
            StringHelper::format(static_cast<uint64_t>(estimatedNumEntities)),
            StringHelper::format(static_cast<uint64_t>(MaxObjectsPerCommand))));
    }
}

void McpMultiplierTools::storeForUndo(MultiplierService::Result&& result)
{
    _origSelection = std::move(result.origSelection);
    _selectionAfterMultiplication = _SimulationFacade::get()->getSelectionShallowData();
}

std::string McpMultiplierTools::describeSelection() const
{
    auto selection = _SimulationFacade::get()->getSelectionShallowData();
    return std::format(
        "The selection now contains {} objects and {} energy particles.",
        StringHelper::format(static_cast<uint64_t>(selection.numClusterCells)),
        StringHelper::format(static_cast<uint64_t>(selection.numEnergyParticles)));
}
