#include "McpInspectionTools.h"

#include <algorithm>
#include <ranges>
#include <set>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Base/Math.h>

#include <Data/DescEditService.h>
#include <Data/DescValidationService.h>

#include <EngineInterface/InspectedEntityIds.h>
#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/SerializerService.h>

#include <Network/McpArguments.h>
#include <Network/McpJson.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr DefaultSearchRadius = 5.0f;
    auto constexpr MaxSearchRadius = 200.0f;
    auto constexpr DefaultMaxResults = 50;
    auto constexpr MaxResults = 500;
    auto constexpr MaxInspectedObjects = Const::MaxInspectedObjects;

    auto constexpr JsonConventions =
        "JSON conventions: ids are strings. Fields with default values are omitted unless 'complete' is set. A field that holds one of several "
        "alternatives (e.g. the cell type) is an object with exactly one field named after the alternative, e.g. {\"depot\": {...}}. Enumerations "
        "are integers. Positions and velocities are [x, y] arrays.";

    std::string getFirstKey(boost::json::value const& json)
    {
        if (!json.is_object() || json.as_object().empty()) {
            return {};
        }
        return std::string(json.as_object().begin()->key());
    }

    RealVector2D getPos(ExtendedObjectOrEnergyDesc const& entity)
    {
        if (std::holds_alternative<ExtendedObjectDesc>(entity)) {
            return std::get<ExtendedObjectDesc>(entity).object._pos;
        }
        return std::get<EnergyDesc>(entity)._pos;
    }

    boost::json::array toJson(RealVector2D const& value)
    {
        return {value.x, value.y};
    }
}

std::vector<McpTool> McpInspectionTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "find_objects",
            .description = "Finds the objects and energy particles near a position and returns a short summary of each, nearest first. Like a "
                           "selection in ALIEN, it replaces the current selection by the searched square area. Use inspect_objects for details.",
            .inputSchema = McpSchema::object(
                {
                    {"x", McpSchema::number("X coordinate")},
                    {"y", McpSchema::number("Y coordinate")},
                    {"radius", McpSchema::number(std::format("Search radius, default: {}", DefaultSearchRadius), 0.1, MaxSearchRadius)},
                    {"max_results", McpSchema::integer(std::format("Maximum number of results, default: {}", DefaultMaxResults), 1, MaxResults)},
                },
                {"x", "y"}),
            .handler = [this](boost::json::object const& arguments) { return findObjects(arguments); },
        },
        McpTool{
            .name = "inspect_objects",
            .description = std::string("Returns all properties of objects or energy particles: position, velocity, connections, type-specific "
                                       "properties such as the cell type, energies, neural network and signals, and for cells of creatures the "
                                       "creature properties and optionally the genome. ")
                + JsonConventions,
            .inputSchema = McpSchema::object(
                {
                    {"ids", McpSchema::ids(std::format("Ids of the objects or energy particles, at most {}", MaxInspectedObjects), 1)},
                    {"include_genomes", McpSchema::boolean("Also return the genomes of the creatures. Default: false")},
                    {"complete", McpSchema::boolean("Also return fields with default values. Default: false")},
                },
                {"ids"}),
            .handler = [this](boost::json::object const& arguments) { return inspectObjects(arguments); },
        },
        McpTool{
            .name = "change_object",
            .description = std::string("Changes properties of an object or energy particle like the inspection window of ALIEN. Only the given "
                                       "fields are changed; the JSON has the same structure as returned by inspect_objects. Invalid values are "
                                       "corrected. Ids cannot be changed. ")
                + JsonConventions,
            .inputSchema = McpSchema::object(
                {
                    {"id", McpSchema::id("Id of the object or energy particle")},
                    {"object", McpSchema::jsonObject("Fields of the object or energy particle to change, e.g. {\"color\": 2}")},
                    {"creature", McpSchema::jsonObject("Fields of the creature of a cell to change, e.g. {\"generation\": 5}")},
                },
                {"id"}),
            .handler = [this](boost::json::object const& arguments) { return changeObject(arguments); },
        },
        McpTool{
            .name = "get_json_format",
            .description = std::string("Describes the JSON format of objects or genomes by an example with all fields and their default values. "
                                       "Where one of several alternatives can be chosen, all alternatives are listed under 'oneOf'. Arrays show one "
                                       "example element. ")
                + JsonConventions,
            .inputSchema = McpSchema::object({{"kind", McpSchema::enumeration("Kind of description", {"object", "genome"})}}, {"kind"}),
            .handler = [this](boost::json::object const& arguments) { return getJsonFormat(arguments); },
        },
    };
}

McpToolResult McpInspectionTools::findObjects(boost::json::object const& arguments) const
{
    RealVector2D center{McpArguments::getFloat(arguments, "x"), McpArguments::getFloat(arguments, "y")};
    auto radius = McpArguments::getOptionalFloat(arguments, "radius", 0.1f, MaxSearchRadius).value_or(DefaultSearchRadius);
    auto maxResults = McpArguments::getOptionalInt(arguments, "max_results", 1, MaxResults).value_or(DefaultMaxResults);

    _SimulationFacade::get()->setSelection({center.x - radius, center.y - radius}, {center.x + radius, center.y + radius});
    _context->onSelectionChanged();
    auto entities = DescEditService::get().getObjects(_SimulationFacade::get()->getSelectedSimulationData(false));

    std::erase_if(entities, [&](auto const& entity) { return Math::length(getPos(entity) - center) > radius; });
    std::ranges::sort(entities, {}, [&](auto const& entity) { return Math::length(getPos(entity) - center); });

    boost::json::array results;
    for (auto const& entity : entities | std::views::take(maxResults)) {
        results.emplace_back(describeBriefly(entity, center));
    }
    return {.text = McpJson::serialize(boost::json::object{{"found", entities.size()}, {"results", std::move(results)}})};
}

McpToolResult McpInspectionTools::inspectObjects(boost::json::object const& arguments) const
{
    auto ids = McpArguments::getIds(arguments, "ids", 1);
    if (ids.size() > MaxInspectedObjects) {
        throw std::invalid_argument(std::format("At most {} objects can be inspected at once.", MaxInspectedObjects));
    }
    auto includeGenomes = McpArguments::getOptionalBool(arguments, "include_genomes").value_or(false);
    auto omitDefaultValues = !McpArguments::getOptionalBool(arguments, "complete").value_or(false);
    auto const& serializer = SerializerService::get();

    auto entities = DescEditService::get().getObjects(_SimulationFacade::get()->getInspectedSimulationData(ids));

    boost::json::array objects;
    boost::json::array energyParticles;
    boost::json::object genomes;
    std::set<uint64_t> foundIds;
    for (auto const& entity : entities) {
        foundIds.insert(DescEditService::get().getId(entity));
        if (std::holds_alternative<EnergyDesc>(entity)) {
            energyParticles.emplace_back(serializer.serializeToJson(std::get<EnergyDesc>(entity), omitDefaultValues));
            continue;
        }
        auto const& extendedObject = std::get<ExtendedObjectDesc>(entity);
        boost::json::object entry{{"object", serializer.serializeToJson(extendedObject.object, omitDefaultValues)}};
        if (extendedObject.creature) {
            entry["creature"] = serializer.serializeToJson(*extendedObject.creature, omitDefaultValues);
        }
        if (includeGenomes && extendedObject.genome) {
            genomes[std::to_string(extendedObject.genome->_id)] = serializer.serializeToJson(*extendedObject.genome, omitDefaultValues);
        }
        objects.emplace_back(std::move(entry));
    }

    boost::json::object result{{"objects", std::move(objects)}, {"energy_particles", std::move(energyParticles)}};
    if (includeGenomes) {
        result["genomes_by_id"] = std::move(genomes);
    }
    boost::json::array missingIds;
    for (auto const& id : ids) {
        if (!foundIds.contains(id)) {
            missingIds.emplace_back(std::to_string(id));
        }
    }
    if (!missingIds.empty()) {
        result["not_found_ids"] = std::move(missingIds);
    }
    return {.text = McpJson::serialize(result)};
}

McpToolResult McpInspectionTools::changeObject(boost::json::object const& arguments) const
{
    auto id = McpArguments::getId(arguments, "id");
    auto objectPatch = arguments.if_contains("object");
    auto creaturePatch = arguments.if_contains("creature");
    if (!objectPatch && !creaturePatch) {
        throw std::invalid_argument("Give the fields to change in 'object' or 'creature'.");
    }
    auto entity = getInspectedEntity(id);
    auto const& serializer = SerializerService::get();

    if (std::holds_alternative<EnergyDesc>(entity)) {
        if (creaturePatch) {
            throw std::invalid_argument("An energy particle has no creature.");
        }
        auto& energy = std::get<EnergyDesc>(entity);
        serializer.deserializeFromJson(energy, *objectPatch, true);
        if (energy._id != id) {
            throw std::invalid_argument("The id cannot be changed.");
        }
        _SimulationFacade::get()->changeParticle(energy);
    } else {
        auto& extendedObject = std::get<ExtendedObjectDesc>(entity);
        if (objectPatch) {
            serializer.deserializeFromJson(extendedObject.object, *objectPatch, true);
        }
        if (creaturePatch) {
            if (!extendedObject.creature) {
                throw std::invalid_argument("The object does not belong to a creature.");
            }
            auto creatureId = extendedObject.creature->_id;
            serializer.deserializeFromJson(*extendedObject.creature, *creaturePatch, true);
            if (extendedObject.creature->_id != creatureId) {
                throw std::invalid_argument("The id of the creature cannot be changed.");
            }
        }
        if (extendedObject.object._id != id) {
            throw std::invalid_argument("The id cannot be changed.");
        }
        DescValidationService::get().validateAndCorrect(extendedObject);
        _SimulationFacade::get()->changeCell(extendedObject);
    }
    _context->onSelectionChanged();

    auto changedEntity = getInspectedEntity(id);
    boost::json::object result;
    if (std::holds_alternative<EnergyDesc>(changedEntity)) {
        result["energy_particle"] = serializer.serializeToJson(std::get<EnergyDesc>(changedEntity));
    } else {
        auto const& extendedObject = std::get<ExtendedObjectDesc>(changedEntity);
        result["object"] = serializer.serializeToJson(extendedObject.object);
        if (extendedObject.creature) {
            result["creature"] = serializer.serializeToJson(*extendedObject.creature);
        }
    }
    return {.text = McpJson::serialize(result)};
}

McpToolResult McpInspectionTools::getJsonFormat(boost::json::object const& arguments) const
{
    auto kind = McpArguments::getString(arguments, "kind");
    if (kind == "genome") {
        return {.text = McpJson::serialize(SerializerService::get().getJsonFormatOfGenome())};
    }
    if (kind == "object") {
        return {.text = McpJson::serialize(SerializerService::get().getJsonFormatOfObject())};
    }
    throw std::invalid_argument("'kind' must be 'object' or 'genome'.");
}

boost::json::object McpInspectionTools::describeBriefly(ExtendedObjectOrEnergyDesc const& entity, RealVector2D const& center) const
{
    auto pos = getPos(entity);
    boost::json::object result{
        {"id", std::to_string(DescEditService::get().getId(entity))},
        {"position", toJson(pos)},
        {"distance", Math::length(pos - center)},
    };
    if (std::holds_alternative<EnergyDesc>(entity)) {
        auto const& energy = std::get<EnergyDesc>(entity);
        result["kind"] = "energyParticle";
        result["energy"] = energy._energy;
        result["color"] = energy._color;
        return result;
    }

    auto const& extendedObject = std::get<ExtendedObjectDesc>(entity);
    auto const& object = extendedObject.object;
    auto objectJson = SerializerService::get().serializeToJson(object);
    auto const& typeJson = objectJson.as_object().at("type");
    auto kind = getFirstKey(typeJson);
    result["kind"] = kind;
    result["color"] = object._color;
    result["connections"] = object._connections.size();
    std::visit(
        [&](auto const& type) {
            using Type = std::decay_t<decltype(type)>;
            if constexpr (std::same_as<Type, CellDesc>) {
                result["energy"] = type._usableEnergy;
                result["cell_type"] = getFirstKey(typeJson.as_object().at(kind).as_object().at("cellType"));
            } else {
                result["energy"] = type._energy;
            }
        },
        object._type);
    if (extendedObject.creature) {
        result["creature_id"] = std::to_string(extendedObject.creature->_id);
    }
    return result;
}

ExtendedObjectOrEnergyDesc McpInspectionTools::getInspectedEntity(uint64_t id) const
{
    auto entities = DescEditService::get().getObjects(_SimulationFacade::get()->getInspectedSimulationData({id}));
    auto findResult = std::ranges::find_if(entities, [&](auto const& entity) { return DescEditService::get().getId(entity) == id; });
    if (findResult == entities.end()) {
        throw std::invalid_argument(std::format("There is no object or energy particle with id {}.", id));
    }
    return *findResult;
}
