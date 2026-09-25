#include "McpParameterTools.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <ranges>
#include <stdexcept>
#include <cctype>
#include <format>
#include <span>

#include <boost/algorithm/string.hpp>
#include <boost/json.hpp>
#include <boost/range/adaptors.hpp>

#include <Fonts/IconsFontAwesome5.h>

#include <Base/StringHelper.h>

#include <EngineInterface/LocationHelper.h>
#include <EngineInterface/ParametersAccessService.h>
#include <EngineInterface/ParametersEditService.h>
#include <EngineInterface/ParametersValidationService.h>
#include <EngineInterface/SimulationFacade.h>
#include <EngineInterface/SpecificationEvaluationService.h>

#include <PersisterInterface/SerializerService.h>

#include <Network/McpArguments.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr SettingsFileExtension = ".settings.json";

    std::vector<std::string> getExpertGroupNames()
    {
        std::vector<std::string> result;
        for (auto const& groupSpec : SimulationParameters::getSpec()._groups) {
            if (groupSpec._expertToggle) {
                result.emplace_back("'" + groupSpec._name + "'");
            }
        }
        return result;
    }

    boost::json::object withLocation(boost::json::object properties)
    {
        properties["location"] =
            McpSchema::integer("Location number from list_locations: 0 = base parameters (default), higher numbers = layers and radiation sources", 0);
        return properties;
    }
}

std::vector<McpTool> McpParameterTools::getTools()
{
    return {
        McpTool{
            .name = "list_parameter_groups",
            .description = "Lists the groups of simulation parameters that are available at a location together with their number of parameters. "
                           "Start here to explore the simulation parameters.",
            .inputSchema = McpSchema::object(withLocation({})),
            .handler = [this](boost::json::object const& arguments) { return listParameterGroups(arguments); },
        },
        McpTool{
            .name = "get_parameters",
            .description = "Returns the simulation parameters of a group at a location with their paths, types, current values and valid ranges. "
                           "'colors' tells whether a parameter has one value per color or a color matrix (row = color, column = target color). "
                           "'enabled' (if present) tells whether the parameter is active; at layers it tells whether the layer overrides the base value.",
            .inputSchema = McpSchema::object(
                withLocation({
                    {"group", McpSchema::string("Name of the parameter group, see list_parameter_groups")},
                    {"include_descriptions", McpSchema::boolean("Also return the description of each parameter. Default: false")},
                }),
                {"group"}),
            .handler = [this](boost::json::object const& arguments) { return getParameters(arguments); },
        },
        McpTool{
            .name = "set_parameters",
            .description =
                "Changes simulation parameters. All changes are applied together; if one of them is invalid, nothing is changed. Values outside the valid "
                "range are corrected and marked as adjusted.\n"
                "Value formats by type: bool, integer, number, vector [x, y], text, option (name of the option), color \"#rrggbb\", "
                "color_transition_rule {\"target_color\": 0-9, \"duration\": integer}. Parameters with infinity_allowed also accept \"infinity\".\n"
                "Parameters with one value per color take a single value for all colors, an array with one value per color, or a single value together "
                "with 'color'. Color matrices take a single value for all entries, an array of rows, a row together with 'color', or a single value "
                "together with 'color' and 'target_color'.\n"
                "Setting a value at a layer lets the layer override the base value unless 'enabled' is given.",
            .inputSchema = McpSchema::object(
                {
                    {"changes",
                     McpSchema::array(
                         "Changes to apply",
                         McpSchema::object(
                             withLocation({
                                 {"path", McpSchema::string("Path of the parameter as returned by get_parameters, e.g. 'Physics: Motion.Friction'")},
                                 {"value", McpSchema::any("New value, see the tool description for the format")},
                                 {"color", McpSchema::integer("Only change the value of this color", 0, MAX_COLORS - 1)},
                                 {"target_color", McpSchema::integer("Only change this column of a color matrix, requires 'color'", 0, MAX_COLORS - 1)},
                                 {"enabled", McpSchema::boolean("Activates or deactivates the parameter or the override by a layer")},
                                 {"pinned", McpSchema::boolean("Pins a relative radiation strength so that it is kept when other strengths change")},
                             }),
                             {"path"}),
                         1)},
                },
                {"changes"}),
            .handler = [this](boost::json::object const& arguments) { return setParameters(arguments); },
        },
        McpTool{
            .name = "enable_expert_settings",
            .description = "Activates or deactivates a group of expert settings. Groups with expert settings: "
                + boost::algorithm::join(getExpertGroupNames(), ", ") + ".",
            .inputSchema = McpSchema::object(
                {
                    {"group", McpSchema::string("Name of the parameter group")},
                    {"enabled", McpSchema::boolean("true to activate, false to deactivate")},
                },
                {"group", "enabled"}),
            .handler = [this](boost::json::object const& arguments) { return enableExpertSettings(arguments); },
        },
        McpTool{
            .name = "reset_parameters",
            .description = "Resets the simulation parameters to their reference values, i.e. the values that the revert buttons in ALIEN restore. These are "
                           "usually the values when the simulation or the parameter file was loaded.",
            .inputSchema = McpSchema::object({{"group", McpSchema::string("Only reset this parameter group at all locations. Default: all parameters")}}),
            .handler = [this](boost::json::object const& arguments) { return resetParameters(arguments); },
        },
        McpTool{
            .name = "list_locations",
            .description = "Lists the locations of the simulation parameters: the base parameters (location 0), the layers and the radiation sources. "
                           "Layers override the base parameters in an area and can exert force fields. Radiation sources emit the energy particles "
                           "that cells radiate. The location numbers are used by the other parameter tools.",
            .inputSchema = McpSchema::object(),
            .handler = [this](boost::json::object const&) { return listLocations(); },
        },
        McpTool{
            .name = "add_layer",
            .description = "Adds a layer with default values. Adjust it afterwards with set_parameters, e.g. 'Location.Position (x,y)', 'Shape.Shape' or "
                           "'Force field.Field type'. The locations behind the new one are renumbered.",
            .inputSchema =
                McpSchema::object({{"after_location", McpSchema::integer("The new layer is inserted behind this location. Default: at the end", 0)}}),
            .handler = [this](boost::json::object const& arguments) { return addLocation(arguments, LocationType::Layer); },
        },
        McpTool{
            .name = "add_radiation_source",
            .description = "Adds a radiation source with default values. Adjust it afterwards with set_parameters, e.g. 'Location.Position (x,y)', "
                           "'Shape.Shape' or 'General.Relative strength'. The locations behind the new one are renumbered.",
            .inputSchema = McpSchema::object(
                {{"after_location", McpSchema::integer("The new radiation source is inserted behind this location. Default: at the end", 0)}}),
            .handler = [this](boost::json::object const& arguments) { return addLocation(arguments, LocationType::Source); },
        },
        McpTool{
            .name = "clone_location",
            .description = "Duplicates a layer or radiation source. The copy is inserted directly behind it and the locations behind it are renumbered.",
            .inputSchema = McpSchema::object({{"location", McpSchema::integer("Location number of the layer or radiation source", 1)}}, {"location"}),
            .handler = [this](boost::json::object const& arguments) { return cloneLocation(arguments); },
        },
        McpTool{
            .name = "delete_location",
            .description = "Deletes a layer or radiation source. The locations behind it are renumbered.",
            .inputSchema = McpSchema::object({{"location", McpSchema::integer("Location number of the layer or radiation source", 1)}}, {"location"}),
            .handler = [this](boost::json::object const& arguments) { return deleteLocation(arguments); },
        },
        McpTool{
            .name = "move_location",
            .description = "Moves a layer or radiation source one position up or down in the order. Later layers take precedence over earlier ones.",
            .inputSchema = McpSchema::object(
                {
                    {"location", McpSchema::integer("Location number of the layer or radiation source", 1)},
                    {"direction", McpSchema::enumeration("Direction of the move", {"up", "down"})},
                },
                {"location", "direction"}),
            .handler = [this](boost::json::object const& arguments) { return moveLocation(arguments); },
        },
        McpTool{
            .name = "load_parameters",
            .description = "Loads simulation parameters from a settings file (*.settings.json) including layers and radiation sources. They also become "
                           "the new reference values.",
            .inputSchema = McpSchema::object({{"file_path", McpSchema::string("Absolute path of the file")}}, {"file_path"}),
            .handler = [this](boost::json::object const& arguments) { return loadParameters(arguments); },
        },
        McpTool{
            .name = "save_parameters",
            .description = "Saves the simulation parameters including layers and radiation sources to a settings file (*.settings.json).",
            .inputSchema = McpSchema::object(
                {
                    {"file_path", McpSchema::string("Absolute path of the file, must end with .settings.json")},
                    {"overwrite", McpSchema::boolean("Overwrite an existing file. Default: false")},
                },
                {"file_path"}),
            .handler = [this](boost::json::object const& arguments) { return saveParameters(arguments); },
        },
    };
}

namespace
{
    int getNumLocations(SimulationParameters const& parameters)
    {
        return 1 + parameters.numLayers + parameters.numSources;
    }

    int getOptionalLocation(boost::json::object const& arguments, SimulationParameters const& parameters)
    {
        return McpArguments::getOptionalInt(arguments, "location", 0, getNumLocations(parameters) - 1).value_or(0);
    }

    std::string getLocationTypeName(LocationType locationType)
    {
        if (locationType == LocationType::Base) {
            return "base";
        } else if (locationType == LocationType::Layer) {
            return "layer";
        } else {
            return "radiation_source";
        }
    }

    std::string getLocationName(SimulationParameters const& parameters, int orderNumber)
    {
        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        if (locationType == LocationType::Base) {
            return "Base";
        }
        auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
        return locationType == LocationType::Layer ? parameters.layerName.layerValues[index] : parameters.sourceName.sourceValues[index];
    }

    ParameterGroupSpec const& findGroup(std::string const& name)
    {
        if (auto result = ParametersAccessService::get().findGroup(name)) {
            return *result;
        }
        throw std::invalid_argument(std::format("Unknown parameter group '{}'. See list_parameter_groups.", name));
    }

    double toJsonNumber(float value)
    {
        return std::stod(std::format("{:.6g}", value));
    }

    boost::json::array toJson(RealVector2D const& value)
    {
        return {toJsonNumber(value.x), toJsonNumber(value.y)};
    }

    bool isInfinityAllowed(ParameterEntry const& entry)
    {
        auto const& reference = entry.spec->_reference;
        if (auto intSpec = std::get_if<IntSpec>(&reference)) {
            return intSpec->_infinity;
        }
        if (auto floatSpec = std::get_if<FloatSpec>(&reference)) {
            return floatSpec->_infinity;
        }
        return false;
    }

    boost::json::value toJson(ParameterScalar const& scalar, bool infinityAllowed)
    {
        if (auto value = std::get_if<bool>(&scalar)) {
            return *value;
        } else if (auto value = std::get_if<int>(&scalar)) {
            return infinityAllowed && *value == Infinity<int>::value ? boost::json::value("infinity") : boost::json::value(*value);
        } else if (auto value = std::get_if<float>(&scalar)) {
            return infinityAllowed && *value >= Infinity<float>::value ? boost::json::value("infinity") : boost::json::value(toJsonNumber(*value));
        } else if (auto value = std::get_if<RealVector2D>(&scalar)) {
            return toJson(*value);
        } else if (auto value = std::get_if<std::string>(&scalar)) {
            return boost::json::value(*value);
        } else if (auto value = std::get_if<FloatColorRGB>(&scalar)) {
            return boost::json::value(StringHelper::formatHexColor(*value));
        } else {
            auto const& rule = std::get<ColorTransitionRule>(scalar);
            return boost::json::object{
                {"target_color", rule.targetColor},
                {"duration", rule.duration == Infinity<int>::value ? boost::json::value("infinity") : boost::json::value(rule.duration)},
            };
        }
    }

    boost::json::value toJson(ParameterValue const& value, bool infinityAllowed)
    {
        if (value.colorDependence == ColorDependence::None) {
            return toJson(value.values.front(), infinityAllowed);
        }
        boost::json::array result;
        if (value.colorDependence == ColorDependence::ColorVector) {
            for (auto const& scalar : value.values) {
                result.emplace_back(toJson(scalar, infinityAllowed));
            }
        } else {
            for (int row = 0; row < MAX_COLORS; ++row) {
                boost::json::array rowValues;
                for (int column = 0; column < MAX_COLORS; ++column) {
                    rowValues.emplace_back(toJson(value.values.at(row * MAX_COLORS + column), infinityAllowed));
                }
                result.emplace_back(std::move(rowValues));
            }
        }
        return result;
    }

    std::string getTypeName(ParameterType type)
    {
        switch (type) {
        case ParameterType::Bool:
            return "bool";
        case ParameterType::Int:
            return "integer";
        case ParameterType::Float:
            return "number";
        case ParameterType::Float2:
            return "vector";
        case ParameterType::Text:
            return "text";
        case ParameterType::Alternative:
            return "option";
        case ParameterType::Color:
            return "color";
        default:
            return "color_transition_rule";
        }
    }

    void addValueRange(boost::json::object& result, ParameterEntry const& entry, IntVector2D const& worldSize)
    {
        auto const& reference = entry.spec->_reference;
        if (auto intSpec = std::get_if<IntSpec>(&reference)) {
            result["min"] = intSpec->_min;
            result["max"] = intSpec->_max;
        } else if (auto floatSpec = std::get_if<FloatSpec>(&reference)) {
            result["min"] = toJsonNumber(std::get<float>(floatSpec->_min));
            result["max"] = std::holds_alternative<MaxWorldRadiusSize>(floatSpec->_max) ? std::max(worldSize.x, worldSize.y)
                                                                                        : toJsonNumber(std::get<float>(floatSpec->_max));
        } else if (auto float2Spec = std::get_if<Float2Spec>(&reference)) {
            result["min"] = toJson(std::get<RealVector2D>(float2Spec->_min));
            result["max"] =
                std::holds_alternative<WorldSize>(float2Spec->_max) ? toJson(toRealVector2D(worldSize)) : toJson(std::get<RealVector2D>(float2Spec->_max));
        } else if (auto alternativeSpec = std::get_if<AlternativeSpec>(&reference)) {
            boost::json::array options;
            for (auto const& name : alternativeSpec->_alternatives | std::views::keys) {
                options.emplace_back(name);
            }
            result["options"] = std::move(options);
        }
        if (isInfinityAllowed(entry)) {
            result["infinity_allowed"] = true;
        }
    }

    boost::json::object toJson(ParameterEntry const& entry, SimulationParameters const& parameters, int orderNumber, bool includeDescription)
    {
        auto& service = ParametersAccessService::get();
        auto value = service.getValue(entry, parameters, orderNumber);

        boost::json::object result{
            {"path", entry.path},
            {"type", getTypeName(service.getType(entry))},
            {"value", toJson(value, isInfinityAllowed(entry))},
        };
        if (value.colorDependence == ColorDependence::ColorVector) {
            result["colors"] = "per_color";
        } else if (value.colorDependence == ColorDependence::ColorMatrix) {
            result["colors"] = "color_matrix";
        }
        if (value.enabled.has_value()) {
            result["enabled"] = value.enabled.value();
        }
        if (value.pinned.has_value()) {
            result["pinned"] = value.pinned.value();
        }
        addValueRange(result, entry, _SimulationFacade::get()->getWorldSize());
        if (includeDescription && entry.spec->_description.has_value()) {
            result["description"] = boost::algorithm::replace_all_copy(entry.spec->_description.value(), ICON_FA_CHEVRON_RIGHT, "-");
        }
        return result;
    }
}

McpToolResult McpParameterTools::listParameterGroups(boost::json::object const& arguments) const
{
    auto& service = ParametersAccessService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto orderNumber = getOptionalLocation(arguments, parameters);
    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);

    boost::json::array groups;
    for (auto const& groupSpec : service.getGroups(locationType)) {
        boost::json::object group{{"name", groupSpec->_name}, {"parameters", service.getParameters(*groupSpec, parameters, orderNumber).size()}};
        if (groupSpec->_expertToggle) {
            group["expert_settings_enabled"] = *SpecificationEvaluationService::get().getExpertToggleRef(groupSpec->_expertToggle, parameters);
        }
        if (groupSpec->_description.has_value()) {
            group["description"] = groupSpec->_description.value();
        }
        groups.emplace_back(std::move(group));
    }
    auto result = boost::json::object{
        {"location", orderNumber},
        {"location_type", getLocationTypeName(locationType)},
        {"groups", std::move(groups)},
    };
    return {.text = boost::json::serialize(result)};
}

McpToolResult McpParameterTools::getParameters(boost::json::object const& arguments) const
{
    auto& service = ParametersAccessService::get();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto const& groupSpec = findGroup(McpArguments::getString(arguments, "group"));
    auto orderNumber = getOptionalLocation(arguments, parameters);
    auto includeDescriptions = McpArguments::getOptionalBool(arguments, "include_descriptions").value_or(false);

    auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
    if (!SpecificationEvaluationService::get().isVisible(groupSpec, locationType)) {
        throw std::invalid_argument(
            std::format("The group '{}' is not available for location {} ({}).", groupSpec._name, orderNumber, getLocationTypeName(locationType)));
    }

    boost::json::array entries;
    for (auto const& entry : service.getParameters(groupSpec, parameters, orderNumber)) {
        entries.emplace_back(toJson(entry, parameters, orderNumber, includeDescriptions));
    }
    auto result = boost::json::object{
        {"group", groupSpec._name},
        {"location", orderNumber},
        {"location_type", getLocationTypeName(locationType)},
    };
    if (groupSpec._expertToggle) {
        result["expert_settings_enabled"] = *SpecificationEvaluationService::get().getExpertToggleRef(groupSpec._expertToggle, parameters);
    }
    result["parameters"] = std::move(entries);
    return {.text = boost::json::serialize(result)};
}

namespace
{
    ParameterEntry findParameter(std::string const& path, SimulationParameters const& parameters, int orderNumber)
    {
        auto& service = ParametersAccessService::get();
        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        if (auto result = service.findParameter(path, locationType)) {
            return result.value();
        }

        std::vector<std::string> otherLocationTypes;
        for (auto const& otherLocationType : {LocationType::Base, LocationType::Layer, LocationType::Source}) {
            if (otherLocationType != locationType && service.findParameter(path, otherLocationType)) {
                otherLocationTypes.emplace_back(getLocationTypeName(otherLocationType));
            }
        }
        if (!otherLocationTypes.empty()) {
            throw std::invalid_argument(std::format(
                "The parameter '{}' is not available for location {} ({}), only for: {}.",
                path,
                orderNumber,
                getLocationTypeName(locationType),
                boost::algorithm::join(otherLocationTypes, ", ")));
        }
        throw std::invalid_argument(std::format("Unknown parameter '{}'. Paths have the form '<group>.<parameter>', see get_parameters.", path));
    }

    std::invalid_argument createInvalidValueError(ParameterEntry const& entry, std::string const& expected)
    {
        return std::invalid_argument(std::format("The value for '{}' must be {}.", entry.path, expected));
    }

    bool isInfinity(boost::json::value const& json)
    {
        return json.is_string() && boost::algorithm::iequals(std::string(json.as_string()), "infinity");
    }

    double toNumber(boost::json::value const& json, ParameterEntry const& entry, std::string const& expected)
    {
        if (!json.is_number()) {
            throw createInvalidValueError(entry, expected);
        }
        return json.to_number<double>();
    }

    int toIntValue(boost::json::value const& json, ParameterEntry const& entry, bool infinityAllowed)
    {
        if (infinityAllowed && isInfinity(json)) {
            return Infinity<int>::value;
        }
        auto expected = infinityAllowed ? "an integer or \"infinity\"" : "an integer";
        auto number = toNumber(json, entry, expected);
        if (number != std::floor(number) || number < std::numeric_limits<int>::lowest() || number > std::numeric_limits<int>::max()) {
            throw createInvalidValueError(entry, expected);
        }
        return static_cast<int>(number);
    }

    FloatColorRGB toColor(boost::json::value const& json, ParameterEntry const& entry)
    {
        auto isHexColor = [](std::string const& text) {
            return text.size() == 7 && text.front() == '#' && std::all_of(text.begin() + 1, text.end(), [](unsigned char c) { return std::isxdigit(c); });
        };
        if (!json.is_string() || !isHexColor(std::string(json.as_string()))) {
            throw createInvalidValueError(entry, "a color of the form \"#rrggbb\"");
        }
        auto text = std::string(json.as_string());
        auto getComponent = [&](int index) { return toFloat(std::stoi(text.substr(1 + index * 2, 2), nullptr, 16)) / 255.0f; };
        return {getComponent(0), getComponent(1), getComponent(2)};
    }

    ParameterScalar toScalar(boost::json::value const& json, ParameterEntry const& entry, ParameterScalar const& currentScalar)
    {
        auto type = ParametersAccessService::get().getType(entry);
        auto infinityAllowed = isInfinityAllowed(entry);

        if (type == ParameterType::Bool) {
            if (!json.is_bool()) {
                throw createInvalidValueError(entry, "true or false");
            }
            return json.as_bool();
        } else if (type == ParameterType::Int) {
            return toIntValue(json, entry, infinityAllowed);
        } else if (type == ParameterType::Float) {
            if (infinityAllowed && isInfinity(json)) {
                return Infinity<float>::value;
            }
            return static_cast<float>(toNumber(json, entry, infinityAllowed ? "a number or \"infinity\"" : "a number"));
        } else if (type == ParameterType::Float2) {
            if (!json.is_array() || json.as_array().size() != 2) {
                throw createInvalidValueError(entry, "an array [x, y]");
            }
            auto const& array = json.as_array();
            return RealVector2D{
                static_cast<float>(toNumber(array.at(0), entry, "an array [x, y]")), static_cast<float>(toNumber(array.at(1), entry, "an array [x, y]"))};
        } else if (type == ParameterType::Text || type == ParameterType::Alternative) {
            if (!json.is_string()) {
                throw createInvalidValueError(entry, "a string");
            }
            return std::string(json.as_string());
        } else if (type == ParameterType::Color) {
            return toColor(json, entry);
        } else {
            if (!json.is_object()) {
                throw createInvalidValueError(entry, "an object with 'target_color' and 'duration'");
            }
            auto const& object = json.as_object();
            auto result = std::get<ColorTransitionRule>(currentScalar);
            if (auto targetColor = McpArguments::getOptionalInt(object, "target_color", 0, MAX_COLORS - 1)) {
                result.targetColor = targetColor.value();
            }
            if (auto duration = object.if_contains("duration")) {
                result.duration = toIntValue(*duration, entry, true);
            }
            return result;
        }
    }

    void applyJsonToRange(boost::json::value const& json, ParameterEntry const& entry, std::span<ParameterScalar> scalars)
    {
        if (json.is_array()) {
            auto const& array = json.as_array();
            if (array.size() != scalars.size()) {
                throw std::invalid_argument(std::format("The value for '{}' must be a single value or an array with {} entries.", entry.path, scalars.size()));
            }
            for (auto const& [scalar, element] : std::views::zip(scalars, array)) {
                scalar = toScalar(element, entry, scalar);
            }
        } else {
            for (auto& scalar : scalars) {
                scalar = toScalar(json, entry, scalar);
            }
        }
    }

    void applyJson(boost::json::value const& json, ParameterEntry const& entry, std::optional<int> color, std::optional<int> targetColor, ParameterValue& value)
    {
        auto& scalars = value.values;
        if (value.colorDependence == ColorDependence::None) {
            scalars.front() = toScalar(json, entry, scalars.front());
        } else if (value.colorDependence == ColorDependence::ColorVector) {
            if (color.has_value()) {
                scalars.at(color.value()) = toScalar(json, entry, scalars.at(color.value()));
            } else {
                applyJsonToRange(json, entry, scalars);
            }
        } else {
            if (color.has_value() && targetColor.has_value()) {
                auto& scalar = scalars.at(color.value() * MAX_COLORS + targetColor.value());
                scalar = toScalar(json, entry, scalar);
            } else if (color.has_value()) {
                applyJsonToRange(json, entry, std::span(scalars).subspan(color.value() * MAX_COLORS, MAX_COLORS));
            } else if (json.is_array()) {
                auto const& rows = json.as_array();
                if (rows.size() != MAX_COLORS) {
                    throw std::invalid_argument(std::format("The value for '{}' must be a single value or an array with {} rows.", entry.path, MAX_COLORS));
                }
                for (auto const& [index, row] : rows | boost::adaptors::indexed(0)) {
                    applyJsonToRange(row, entry, std::span(scalars).subspan(index * MAX_COLORS, MAX_COLORS));
                }
            } else {
                applyJsonToRange(json, entry, scalars);
            }
        }
    }

    struct AppliedChange
    {
        ParameterEntry entry;
        int orderNumber = 0;
        ParameterValue requestedValue;
        std::vector<size_t> changedIndices;
    };

    std::vector<size_t> getAddressedIndices(ParameterValue const& value, std::optional<int> color, std::optional<int> targetColor)
    {
        auto indexRange = [](size_t begin, size_t end) {
            auto indices = std::views::iota(begin, end);
            return std::vector<size_t>(indices.begin(), indices.end());
        };
        if (value.colorDependence == ColorDependence::None) {
            return {0};
        }
        if (color.has_value() && targetColor.has_value()) {
            return {static_cast<size_t>(color.value() * MAX_COLORS + targetColor.value())};
        }
        if (color.has_value()) {
            auto isVector = value.colorDependence == ColorDependence::ColorVector;
            auto begin = static_cast<size_t>(isVector ? color.value() : color.value() * MAX_COLORS);
            return indexRange(begin, begin + (isVector ? 1 : MAX_COLORS));
        }
        return indexRange(0, value.values.size());
    }

    AppliedChange applyChange(boost::json::object const& change, SimulationParameters& parameters)
    {
        auto& service = ParametersAccessService::get();
        auto orderNumber = getOptionalLocation(change, parameters);
        auto entry = findParameter(McpArguments::getString(change, "path"), parameters, orderNumber);
        auto value = service.getValue(entry, parameters, orderNumber);

        auto jsonValue = change.if_contains("value");
        auto color = McpArguments::getOptionalInt(change, "color", 0, MAX_COLORS - 1);
        auto targetColor = McpArguments::getOptionalInt(change, "target_color", 0, MAX_COLORS - 1);
        auto enabled = McpArguments::getOptionalBool(change, "enabled");
        auto pinned = McpArguments::getOptionalBool(change, "pinned");

        if (!jsonValue && !enabled.has_value() && !pinned.has_value()) {
            throw std::invalid_argument(std::format("The change of '{}' needs 'value', 'enabled' or 'pinned'.", entry.path));
        }
        if (color.has_value() && value.colorDependence == ColorDependence::None) {
            throw std::invalid_argument(std::format("The parameter '{}' does not depend on colors.", entry.path));
        }
        if (targetColor.has_value() && (!color.has_value() || value.colorDependence != ColorDependence::ColorMatrix)) {
            throw std::invalid_argument(std::format("'target_color' requires 'color' and a color matrix parameter ('{}').", entry.path));
        }

        std::vector<size_t> changedIndices;
        if (jsonValue) {
            applyJson(*jsonValue, entry, color, targetColor, value);
            changedIndices = getAddressedIndices(value, color, targetColor);
            auto isLayer = LocationHelper::getLocationType(orderNumber, parameters) == LocationType::Layer;
            if (isLayer && value.enabled.has_value() && !enabled.has_value()) {
                value.enabled = true;
            }
        }
        if (enabled.has_value()) {
            value.enabled = enabled.value();
        }
        if (pinned.has_value()) {
            value.pinned = pinned.value();
        }
        service.setValue(entry, parameters, orderNumber, value);
        return AppliedChange{
            .entry = entry,
            .orderNumber = orderNumber,
            .requestedValue = service.getValue(entry, parameters, orderNumber),
            .changedIndices = std::move(changedIndices)};
    }

    void addAppliedChange(std::vector<AppliedChange>& appliedChanges, AppliedChange&& newChange)
    {
        for (auto& appliedChange : appliedChanges) {
            if (appliedChange.entry.spec == newChange.entry.spec && appliedChange.orderNumber == newChange.orderNumber) {
                std::erase_if(appliedChange.changedIndices, [&](size_t index) { return std::ranges::contains(newChange.changedIndices, index); });
            }
        }
        appliedChanges.emplace_back(std::move(newChange));
    }

    bool isAdjusted(AppliedChange const& appliedChange, ParameterValue const& value)
    {
        return std::ranges::any_of(
            appliedChange.changedIndices, [&](size_t index) { return value.values.at(index) != appliedChange.requestedValue.values.at(index); });
    }
}

McpToolResult McpParameterTools::setParameters(boost::json::object const& arguments) const
{
    auto& service = ParametersAccessService::get();
    auto changes = McpArguments::getObjects(arguments, "changes", 1);
    auto origParameters = _SimulationFacade::get()->getSimulationParameters();
    auto parameters = origParameters;

    std::vector<AppliedChange> appliedChanges;
    for (auto const& change : changes) {
        addAppliedChange(appliedChanges, applyChange(change, parameters));
    }
    ParametersValidationService::get().validateAndCorrect({_SimulationFacade::get()->getWorldSize()}, parameters);

    auto positionsChanged = parameters.layerPosition != origParameters.layerPosition || parameters.sourcePosition != origParameters.sourcePosition;
    auto updateConfig = positionsChanged || !_SimulationFacade::get()->isSimulationRunning() ? SimulationParametersUpdateConfig::All
                                                                                             : SimulationParametersUpdateConfig::AllExceptChangingPositions;
    _SimulationFacade::get()->setSimulationParameters(parameters, updateConfig);

    boost::json::array changed;
    auto anyAdjusted = false;
    for (auto const& appliedChange : appliedChanges) {
        auto value = service.getValue(appliedChange.entry, parameters, appliedChange.orderNumber);
        boost::json::object entry{
            {"path", appliedChange.entry.path},
            {"location", appliedChange.orderNumber},
            {"value", toJson(value, isInfinityAllowed(appliedChange.entry))},
        };
        if (value.enabled.has_value()) {
            entry["enabled"] = value.enabled.value();
        }
        if (value.pinned.has_value()) {
            entry["pinned"] = value.pinned.value();
        }
        if (isAdjusted(appliedChange, value)) {
            entry["adjusted"] = true;
            anyAdjusted = true;
        }
        changed.emplace_back(std::move(entry));
    }
    auto result = boost::json::object{{"changed", std::move(changed)}};
    if (anyAdjusted) {
        result["note"] = "Values marked as adjusted were corrected to the valid range.";
    }
    return {.text = boost::json::serialize(result)};
}

McpToolResult McpParameterTools::enableExpertSettings(boost::json::object const& arguments) const
{
    auto const& groupSpec = findGroup(McpArguments::getString(arguments, "group"));
    auto enabled = McpArguments::getBool(arguments, "enabled");
    if (!groupSpec._expertToggle) {
        throw std::invalid_argument(std::format(
            "The group '{}' has no expert settings. Groups with expert settings: {}.", groupSpec._name, boost::algorithm::join(getExpertGroupNames(), ", ")));
    }
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    *SpecificationEvaluationService::get().getExpertToggleRef(groupSpec._expertToggle, parameters) = enabled;
    _SimulationFacade::get()->setSimulationParameters(parameters);
    return {.text = std::format("The expert settings '{}' are {}.", groupSpec._name, enabled ? "activated" : "deactivated")};
}

McpToolResult McpParameterTools::resetParameters(boost::json::object const& arguments) const
{
    auto const& origParameters = _SimulationFacade::get()->getOriginalSimulationParameters();
    if (auto groupName = McpArguments::getOptionalString(arguments, "group")) {
        auto const& groupSpec = findGroup(groupName.value());
        auto parameters = _SimulationFacade::get()->getSimulationParameters();
        if (!ParametersAccessService::get().copyGroup(groupSpec, origParameters, parameters)) {
            throw std::invalid_argument("The layers and radiation sources differ from those of the reference values. Only all parameters can be reset.");
        }
        _SimulationFacade::get()->setSimulationParameters(parameters);
        return {.text = std::format("The parameter group '{}' has been reset to its reference values.", groupSpec._name)};
    }
    _SimulationFacade::get()->setSimulationParameters(origParameters);
    return {.text = "All simulation parameters have been reset to their reference values."};
}

McpToolResult McpParameterTools::listLocations() const
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto strengths = ParametersEditService::get().getRadiationStrengths(parameters);

    boost::json::array locations;
    for (int orderNumber = 0; orderNumber < getNumLocations(parameters); ++orderNumber) {
        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        boost::json::object location{
            {"location", orderNumber},
            {"type", getLocationTypeName(locationType)},
            {"name", getLocationName(parameters, orderNumber)},
        };
        if (locationType == LocationType::Base) {
            location["relative_strength"] = toJsonNumber(strengths.values.front());
            location["pinned"] = strengths.pinned.contains(0);
        } else if (locationType == LocationType::Layer) {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            location["position"] = toJson(parameters.layerPosition.layerValues[index]);
            location["velocity"] = toJson(parameters.layerVelocity.layerValues[index]);
            location["opacity"] = toJsonNumber(parameters.layerOpacity.layerValues[index]);
        } else {
            auto index = LocationHelper::findLocationArrayIndex(parameters, orderNumber);
            location["position"] = toJson(parameters.sourcePosition.sourceValues[index]);
            location["velocity"] = toJson(parameters.sourceVelocity.sourceValues[index]);
            location["relative_strength"] = toJsonNumber(strengths.values.at(index + 1));
            location["pinned"] = strengths.pinned.contains(index + 1);
        }
        locations.emplace_back(std::move(location));
    }
    auto result = boost::json::object{
        {"locations", std::move(locations)},
        {"max_layers", MAX_LAYERS},
        {"max_radiation_sources", MAX_SOURCES},
    };
    return {.text = boost::json::serialize(result)};
}

namespace
{
    int getLocation(boost::json::object const& arguments, SimulationParameters const& parameters)
    {
        auto numLocations = getNumLocations(parameters);
        if (numLocations == 1) {
            throw std::invalid_argument("There are no layers or radiation sources.");
        }
        return McpArguments::getInt(arguments, "location", 1, numLocations - 1);
    }

    std::string describeLocation(int orderNumber)
    {
        auto parameters = _SimulationFacade::get()->getSimulationParameters();
        auto locationType = LocationHelper::getLocationType(orderNumber, parameters);
        return std::format(
            "{} '{}' (location {})", locationType == LocationType::Layer ? "layer" : "radiation source", getLocationName(parameters, orderNumber), orderNumber);
    }
}

McpToolResult McpParameterTools::addLocation(boost::json::object const& arguments, LocationType locationType) const
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto numLocations = getNumLocations(parameters);
    auto afterLocation = McpArguments::getOptionalInt(arguments, "after_location", 0, numLocations - 1).value_or(numLocations - 1);

    auto& editService = ParametersEditService::get();
    auto orderNumber = locationType == LocationType::Layer ? editService.insertDefaultLayer(afterLocation) : editService.insertDefaultSource(afterLocation);
    if (!orderNumber.has_value()) {
        throw std::invalid_argument(
            locationType == LocationType::Layer ? "The maximum number of layers has been reached."
                                                : "The maximum number of radiation sources has been reached.");
    }
    return {.text = std::format("Added {}.", describeLocation(orderNumber.value()))};
}

McpToolResult McpParameterTools::cloneLocation(boost::json::object const& arguments) const
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto location = getLocation(arguments, parameters);
    auto sourceDescription = describeLocation(location);

    auto orderNumber = ParametersEditService::get().cloneLocation(location);
    if (!orderNumber.has_value()) {
        throw std::invalid_argument("The maximum number of layers or radiation sources has been reached.");
    }
    return {.text = std::format("Cloned {} to {}.", sourceDescription, describeLocation(orderNumber.value()))};
}

McpToolResult McpParameterTools::deleteLocation(boost::json::object const& arguments) const
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto location = getLocation(arguments, parameters);
    auto description = describeLocation(location);

    ParametersEditService::get().deleteLocation(location);
    return {.text = std::format("Deleted {}. The locations behind it have been renumbered.", description)};
}

McpToolResult McpParameterTools::moveLocation(boost::json::object const& arguments) const
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto location = getLocation(arguments, parameters);
    auto direction = McpArguments::getString(arguments, "direction");
    auto description = describeLocation(location);

    if (direction == "up") {
        if (location == 1) {
            throw std::invalid_argument("The location is already the first one.");
        }
        ParametersEditService::get().moveLocationUpwards(location);
        return {.text = std::format("Moved {} to location {}.", description, location - 1)};
    } else if (direction == "down") {
        if (location == getNumLocations(parameters) - 1) {
            throw std::invalid_argument("The location is already the last one.");
        }
        ParametersEditService::get().moveLocationDownwards(location);
        return {.text = std::format("Moved {} to location {}.", description, location + 1)};
    }
    throw std::invalid_argument("'direction' must be 'up' or 'down'.");
}

McpToolResult McpParameterTools::loadParameters(boost::json::object const& arguments) const
{
    auto filePath = McpArguments::getString(arguments, "file_path");
    SimulationParameters parameters;
    if (!SerializerService::get().deserializeSimulationParametersFromFile(parameters, McpArguments::getFilePath(arguments, "file_path"))) {
        throw std::invalid_argument(std::format("The file '{}' could not be loaded.", filePath));
    }
    _SimulationFacade::get()->setSimulationParameters(parameters);
    _SimulationFacade::get()->setOriginalSimulationParameters(parameters);
    return {
        .text = std::format(
            "Loaded the simulation parameters from '{}' with {} layer(s) and {} radiation source(s).", filePath, parameters.numLayers, parameters.numSources)};
}

McpToolResult McpParameterTools::saveParameters(boost::json::object const& arguments) const
{
    auto filePath = McpArguments::getString(arguments, "file_path");
    auto path = McpArguments::getFilePath(arguments, "file_path");
    if (!filePath.ends_with(SettingsFileExtension)) {
        throw std::invalid_argument(std::format("The file name must end with '{}'.", SettingsFileExtension));
    }
    auto overwrite = McpArguments::getOptionalBool(arguments, "overwrite").value_or(false);
    if (std::filesystem::exists(path) && !overwrite) {
        throw std::invalid_argument(std::format("The file '{}' already exists. Set 'overwrite' to replace it.", filePath));
    }
    if (!SerializerService::get().serializeSimulationParametersToFile(path, _SimulationFacade::get()->getSimulationParameters())) {
        throw std::invalid_argument(std::format("The file '{}' could not be saved.", filePath));
    }
    return {.text = std::format("Saved the simulation parameters to '{}'.", filePath)};
}
