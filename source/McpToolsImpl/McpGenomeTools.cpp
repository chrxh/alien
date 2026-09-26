#include "McpGenomeTools.h"

#include <filesystem>
#include <stdexcept>
#include <format>

#include <boost/json.hpp>

#include <Data/DescEditService.h>
#include <Data/DescValidationService.h>
#include <Data/EngineConstants.h>
#include <Data/GenomeDescAccessService.h>
#include <Data/GenomeDescEditService.h>

#include <EngineInterface/SimulationFacade.h>

#include <PersisterInterface/SerializerService.h>

#include <Network/McpArguments.h>
#include <Network/McpJson.h>
#include <Network/McpSchema.h>

namespace
{
    auto constexpr GenomeFileExtension = ".genome";

    boost::json::object withGenomeSource(boost::json::object properties)
    {
        properties["genome"] = McpSchema::jsonObject(
            "Genome in JSON format, see get_json_format with kind 'genome'. Omitted fields take their default values. Give exactly one of 'genome', "
            "'genome_file_path' and 'genome_of_object_id'.");
        properties["genome_file_path"] = McpSchema::string("Absolute path of a genome file (*.genome)");
        properties["genome_of_object_id"] = McpSchema::id("Id of a cell whose creature's genome is used");
        return properties;
    }
}

std::vector<McpTool> McpGenomeTools::getTools(McpToolContext& context)
{
    _context = &context;

    return {
        McpTool{
            .name = "get_genome",
            .description = "Returns the genome of the creature a cell belongs to in JSON format. The genome consists of genes, each gene of nodes "
                           "describing the cells to construct. See get_json_format with kind 'genome' for all fields.",
            .inputSchema = McpSchema::object(
                {
                    {"object_id", McpSchema::id("Id of a cell of the creature")},
                    {"complete", McpSchema::boolean("Also return fields with default values. Default: false")},
                },
                {"object_id"}),
            .handler = [this](boost::json::object const& arguments) { return getGenome(arguments); },
        },
        McpTool{
            .name = "save_genome",
            .description = "Saves a genome to a genome file (*.genome), which can be opened in the genome editor of ALIEN.",
            .inputSchema = McpSchema::object(
                withGenomeSource({
                    {"file_path", McpSchema::string("Absolute path of the file, must end with .genome")},
                    {"overwrite", McpSchema::boolean("Overwrite an existing file. Default: false")},
                }),
                {"file_path"}),
            .handler = [this](boost::json::object const& arguments) { return saveGenome(arguments); },
        },
        McpTool{
            .name = "create_seed",
            .description = "Places a seed with a genome, i.e. a single constructor cell that builds the creature described by the genome. The "
                           "seed is selected afterwards. The genome is validated and corrected where necessary.",
            .inputSchema = McpSchema::object(withGenomeSource({
                {"x", McpSchema::number("X coordinate, default: center of the visible area")},
                {"y", McpSchema::number("Y coordinate, default: center of the visible area")},
                {"color", McpSchema::integer("Color index of the seed, see get_simulation_info. Default: 0", 0, MAX_COLORS - 1)},
                {"free_energy",
                 McpSchema::boolean("Let the seed construct the creature without consuming its own energy (like 'Create seed with free energy' in "
                                    "ALIEN). Default: false")},
            })),
            .handler = [this](boost::json::object const& arguments) { return createSeed(arguments); },
        },
        McpTool{
            .name = "inject_genome",
            .description = "Replaces the genome of all selected creatures, like the injection in the genome editor of ALIEN.",
            .inputSchema = McpSchema::object(withGenomeSource({})),
            .handler = [this](boost::json::object const& arguments) { return injectGenome(arguments); },
        },
    };
}

McpToolResult McpGenomeTools::getGenome(boost::json::object const& arguments) const
{
    auto genome = getGenomeOfObject(McpArguments::getId(arguments, "object_id"));
    auto complete = McpArguments::getOptionalBool(arguments, "complete").value_or(false);
    return {.text = McpJson::serialize(SerializerService::get().serializeToJson(genome, !complete))};
}

McpToolResult McpGenomeTools::saveGenome(boost::json::object const& arguments) const
{
    auto filePath = McpArguments::getString(arguments, "file_path");
    auto path = McpArguments::getFilePath(arguments, "file_path");
    if (!filePath.ends_with(GenomeFileExtension)) {
        throw std::invalid_argument(std::format("The file name must end with '{}'.", GenomeFileExtension));
    }
    if (std::filesystem::exists(path) && !McpArguments::getOptionalBool(arguments, "overwrite").value_or(false)) {
        throw std::invalid_argument(std::format("The file '{}' already exists. Set 'overwrite' to replace it.", filePath));
    }
    auto genome = getGenomeFromArguments(arguments);
    if (!SerializerService::get().serializeGenomeToFile(path, genome)) {
        throw std::runtime_error(std::format("The genome could not be saved to '{}'.", filePath));
    }
    return {.text = std::format("Saved the genome with {} genes to '{}'.", genome._genes.size(), filePath)};
}

McpToolResult McpGenomeTools::createSeed(boost::json::object const& arguments) const
{
    auto pos = _context->getVisibleAreaCenter();
    pos.x = McpArguments::getOptionalFloat(arguments, "x").value_or(pos.x);
    pos.y = McpArguments::getOptionalFloat(arguments, "y").value_or(pos.y);
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    if (pos.x < 0 || pos.y < 0 || pos.x >= toFloat(worldSize.x) || pos.y >= toFloat(worldSize.y)) {
        throw std::invalid_argument(std::format("The position ({}, {}) lies outside the world.", pos.x, pos.y));
    }
    auto color = McpArguments::getOptionalInt(arguments, "color", 0, MAX_COLORS - 1).value_or(0);
    auto freeEnergy = McpArguments::getOptionalBool(arguments, "free_energy").value_or(false);
    auto genome = getGenomeFromArguments(arguments);
    if (genome._genes.empty()) {
        throw std::invalid_argument("The genome has no genes.");
    }

    auto seed = GenomeDescEditService::get().createSeed(genome, pos, color, freeEnergy);
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(seed));
    _context->onSelectionChanged();
    _context->showMessage("Seed created");

    auto selectedObjects = _SimulationFacade::get()->getSelectedSimulationData(false)._objects;
    auto seedId = selectedObjects.empty() ? std::string("unknown") : std::to_string(selectedObjects.front()._id);
    auto numCells = GenomeDescAccessService::get().getNumberOfResultingCells(genome);
    return {
        .text = std::format(
            "Created the seed with id {} at ({}, {}). Its genome has {} genes and {} nodes, the resulting creature will have {} cells. The "
            "seed is selected.",
            seedId,
            pos.x,
            pos.y,
            genome._genes.size(),
            GenomeDescAccessService::get().getNumberOfNodes(genome),
            numCells < 0 ? std::string("infinitely many") : std::to_string(numCells))};
}

McpToolResult McpGenomeTools::injectGenome(boost::json::object const& arguments) const
{
    auto selection = _SimulationFacade::get()->getSelectionShallowData();
    if (selection.numCreatures == 0) {
        throw std::invalid_argument("No creature is selected. Select creatures with select_area or select_at first.");
    }
    auto genome = getGenomeFromArguments(arguments);
    auto numCreatures = _SimulationFacade::get()->injectGenomeToSelectedCreatures(genome);
    _context->onSelectionChanged();
    _context->showMessage(std::format("Genome injected to {} creatures", numCreatures));
    return {.text = std::format("Injected the genome into {} creatures.", numCreatures)};
}

GenomeDesc McpGenomeTools::getGenomeFromArguments(boost::json::object const& arguments) const
{
    auto numSources = arguments.contains("genome") + arguments.contains("genome_file_path") + arguments.contains("genome_of_object_id");
    if (numSources != 1) {
        throw std::invalid_argument("Give exactly one of 'genome', 'genome_file_path' and 'genome_of_object_id'.");
    }

    GenomeDesc result;
    if (arguments.contains("genome")) {
        SerializerService::get().deserializeFromJson(result, McpArguments::getValue(arguments, "genome"));
    } else if (arguments.contains("genome_file_path")) {
        auto filePath = McpArguments::getString(arguments, "genome_file_path");
        if (!SerializerService::get().deserializeGenomeFromFile(result, McpArguments::getFilePath(arguments, "genome_file_path"))) {
            throw std::invalid_argument(std::format("The genome file '{}' could not be read.", filePath));
        }
    } else {
        result = getGenomeOfObject(McpArguments::getId(arguments, "genome_of_object_id"));
    }
    DescValidationService::get().validateAndCorrect(result);
    return result;
}

GenomeDesc McpGenomeTools::getGenomeOfObject(uint64_t objectId) const
{
    for (auto const& entity : DescEditService::get().getObjects(_SimulationFacade::get()->getInspectedSimulationData({objectId}))) {
        if (!std::holds_alternative<ExtendedObjectDesc>(entity) || DescEditService::get().getId(entity) != objectId) {
            continue;
        }
        auto const& genome = std::get<ExtendedObjectDesc>(entity).genome;
        if (!genome) {
            throw std::invalid_argument(std::format("The object {} does not belong to a creature with a genome.", objectId));
        }
        return *genome;
    }
    throw std::invalid_argument(std::format("There is no object with id {}.", objectId));
}
