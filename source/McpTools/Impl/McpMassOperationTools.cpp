#include "McpMassOperationTools.h"

#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <format>

#include <boost/json.hpp>

#include <Base/Interface/StringHelper.h>

#include <Data/Interface/DescValidationService.h>
#include <Data/Interface/EngineConstants.h>
#include <Data/Interface/GenomeDesc.h>
#include <Data/Interface/MassOperationsService.h>

#include <Engine/Interface/SimulationFacade.h>
#include <Engine/Interface/TemporalControlService.h>

#include <Persister/Interface/SerializerService.h>

#include <Network/Interface/McpArguments.h>
#include <Network/Interface/McpSchema.h>

std::vector<McpTool> McpMassOperationTools::getTools(McpToolContext& context)
{
    _context = &context;

    auto colorList = [](std::string const& description) {
        return McpSchema::array(description, McpSchema::integer("Color index, see get_simulation_info", 0, MAX_COLORS - 1), 1);
    };

    return {
        McpTool{
            .name = "apply_mass_operations",
            .description = "Randomizes properties of many objects at once, like the mass operations dialog in ALIEN. Give at least one operation. Range "
                           "operations need both their minimum and maximum. Randomized values are drawn once per creature and once per connected group of "
                           "non-cell objects, so all objects of such a group share the same value. With restrict_to_selection, the selection including the "
                           "connected cell networks is replaced by the modified objects, which are selected afterwards. Otherwise the whole world is "
                           "modified, which resets the selection.",
            .inputSchema = McpSchema::object({
                {"restrict_to_selection", McpSchema::boolean("Only modify the selection including the connected cell networks. Default: true")},
                {"object_colors", colorList("Randomize the object colors, choosing randomly from these color indices")},
                {"genome_colors", colorList("Randomize the colors of all genome nodes, choosing one color per genome randomly from these color indices")},
                {"min_energy", McpSchema::number("Minimum energy of objects", 0)},
                {"max_energy", McpSchema::number("Maximum energy of objects", 0)},
                {"min_age", McpSchema::integer("Minimum age of cells", 0)},
                {"max_age", McpSchema::integer("Maximum age of cells", 0)},
                {"min_countdown", McpSchema::integer("Minimum countdown of detonator cells", 0)},
                {"max_countdown", McpSchema::integer("Maximum countdown of detonator cells", 0)},
                {"randomize_lineage_ids", McpSchema::boolean("Assign a new random lineage id to each creature. Default: false")},
                {"min_glow", McpSchema::number("Minimum glow of fluid particles", 0, 1)},
                {"max_glow", McpSchema::number("Maximum glow of fluid particles", 0, 1)},
                {"mutation_rates",
                 McpSchema::jsonObject("Mutation rates set for all genomes, in the format of the field 'mutationRates' of a genome, see get_json_format "
                                       "with kind 'genome'. Omitted fields take their default values.")},
            }),
            .handler = [this](boost::json::object const& arguments) { return applyMassOperations(arguments); },
        },
    };
}

namespace
{
    template <typename T>
    struct Range
    {
        T min;
        T max;
    };

    template <typename T>
    std::optional<Range<T>> getOptionalRange(boost::json::object const& arguments, std::string const& minKey, std::string const& maxKey, T maxValue)
    {
        auto getValue = [&](std::string const& key) -> std::optional<T> {
            if constexpr (std::is_same_v<T, int>) {
                return McpArguments::getOptionalInt(arguments, key, 0, maxValue);
            } else {
                return McpArguments::getOptionalFloat(arguments, key, 0, maxValue);
            }
        };
        auto min = getValue(minKey);
        auto max = getValue(maxKey);
        if (!min && !max) {
            return std::nullopt;
        }
        if (!min || !max) {
            throw std::invalid_argument(std::format("'{}' and '{}' must be given together.", minKey, maxKey));
        }
        if (*min > *max) {
            throw std::invalid_argument(std::format("'{}' must not be greater than '{}'.", minKey, maxKey));
        }
        return Range<T>{*min, *max};
    }

    MutationRatesDesc getMutationRates(boost::json::value const& json)
    {
        if (!json.is_object()) {
            throw std::invalid_argument("'mutation_rates' must be a JSON object.");
        }
        GenomeDesc genome;
        SerializerService::get().deserializeFromJson(genome, boost::json::object{{"mutationRates", json}});
        DescValidationService::get().validateAndCorrect(genome);
        return genome._mutationRates;
    }
}

McpToolResult McpMassOperationTools::applyMassOperations(boost::json::object const& arguments) const
{
    auto restrictToSelection = McpArguments::getOptionalBool(arguments, "restrict_to_selection").value_or(true);
    auto objectColors = McpArguments::getOptionalInts(arguments, "object_colors", 1, 0, MAX_COLORS - 1);
    auto genomeColors = McpArguments::getOptionalInts(arguments, "genome_colors", 1, 0, MAX_COLORS - 1);
    auto energies = getOptionalRange(arguments, "min_energy", "max_energy", std::numeric_limits<float>::max());
    auto ages = getOptionalRange(arguments, "min_age", "max_age", std::numeric_limits<int>::max());
    auto countdowns = getOptionalRange(arguments, "min_countdown", "max_countdown", std::numeric_limits<int>::max());
    auto randomizeLineageIds = McpArguments::getOptionalBool(arguments, "randomize_lineage_ids").value_or(false);
    auto glows = getOptionalRange(arguments, "min_glow", "max_glow", 1.0f);
    auto mutationRates = arguments.contains("mutation_rates") ? std::make_optional(getMutationRates(arguments.at("mutation_rates"))) : std::nullopt;

    if (!objectColors && !genomeColors && !energies && !ages && !countdowns && !randomizeLineageIds && !glows && !mutationRates) {
        throw std::invalid_argument("No operation given.");
    }

    if (restrictToSelection) {
        auto selection = _SimulationFacade::get()->getSelectionShallowData();
        if (selection.numObjects == 0 && selection.numEnergyParticles == 0) {
            throw std::invalid_argument("Nothing is selected. Select objects with select_area or select_at first, or set restrict_to_selection to false.");
        }
    }
    auto content = restrictToSelection ? _SimulationFacade::get()->getSelectedSimulationData(true) : _SimulationFacade::get()->getSimulationData();

    auto const& service = MassOperationsService::get();
    std::vector<std::string> operations;
    if (objectColors) {
        service.randomizeCellColors(content, *objectColors);
        operations.emplace_back("object colors");
    }
    if (genomeColors) {
        service.randomizeGenomeColors(content, *genomeColors);
        operations.emplace_back("genome colors");
    }
    if (energies) {
        service.randomizeEnergies(content, energies->min, energies->max);
        operations.emplace_back("energies");
    }
    if (ages) {
        service.randomizeAges(content, ages->min, ages->max);
        operations.emplace_back("ages");
    }
    if (countdowns) {
        service.randomizeCountdowns(content, countdowns->min, countdowns->max);
        operations.emplace_back("detonator countdowns");
    }
    if (randomizeLineageIds) {
        service.randomizeLineageIds(content);
        operations.emplace_back("lineage ids");
    }
    if (glows) {
        service.randomizeGlow(content, glows->min, glows->max);
        operations.emplace_back("fluid glow");
    }
    if (mutationRates) {
        service.setMutationRates(content, *mutationRates);
        operations.emplace_back("mutation rates");
    }

    auto numObjects = content._objects.size();
    auto numGenomes = content._genomes.size();
    if (restrictToSelection) {
        replaceSelection(std::move(content));
    } else {
        replaceWorldContent(content);
    }
    _context->onSelectionChanged();

    std::string operationList;
    for (auto const& operation : operations) {
        operationList += (operationList.empty() ? "" : ", ") + operation;
    }
    return {
        .text = std::format(
            "Modified {} of {} objects and {} genomes in the {}.",
            operationList,
            StringHelper::format(static_cast<uint64_t>(numObjects)),
            StringHelper::format(static_cast<uint64_t>(numGenomes)),
            restrictToSelection ? "selection" : "whole world")};
}

void McpMassOperationTools::replaceSelection(ContentDesc&& content) const
{
    _SimulationFacade::get()->removeSelectedObjects(true);
    _SimulationFacade::get()->addAndSelectSimulationData(std::move(content));
}

void McpMassOperationTools::replaceWorldContent(ContentDesc const& content) const
{
    auto timestep = _SimulationFacade::get()->getCurrentTimestep();
    auto worldSize = _SimulationFacade::get()->getWorldSize();
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto realTime = _SimulationFacade::get()->getRealTime();
    auto statistics = _SimulationFacade::get()->getStatisticsHistory().getCopiedData();
    _SimulationFacade::get()->closeSimulation();

    _SimulationFacade::get()->newSimulation(timestep, worldSize, parameters);
    _SimulationFacade::get()->setSimulationData(content);
    _SimulationFacade::get()->setStatisticsHistory(statistics);
    _SimulationFacade::get()->setRealTime(realTime);
    TemporalControlService::get().createFlashback();
}
