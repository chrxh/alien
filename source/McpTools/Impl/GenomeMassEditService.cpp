#include "GenomeMassEditService.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <ranges>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <concepts>
#include <format>
#include <numbers>

#include <boost/json.hpp>

#include <Base/Definitions.h>
#include <Base/NumberGenerator.h>
#include <Base/StringHelper.h>

#include <Data/DescValidationService.h>
#include <Data/GenomeDescEditService.h>

#include <Persister/Interface/SerializerService.h>

namespace
{
    size_t constexpr MaxExampleChanges = 20;
    size_t constexpr MaxLargestLineages = 5;
    size_t constexpr MaxFrequentValues = 5;
    size_t constexpr MaxDistinctValues = 50;
    size_t constexpr MaxExampleObjects = 5;
}

GenomeMassEditService::GenomeMassEditService()
{
    _genomeFormat = SerializerService::get().getJsonFormatOfGenome();
    auto& genomeFormat = _genomeFormat.as_object();
    _geneFormat = genomeFormat.at("genes").as_array().at(0);
    genomeFormat.erase("genes");
    genomeFormat.erase("id");

    auto& geneFormat = _geneFormat.as_object();
    _nodeFormat = geneFormat.at("nodes").as_array().at(0);
    geneFormat.erase("nodes");

    for (auto const& [name, alternative] : _nodeFormat.at("cellType").at("oneOf").as_object()) {
        _cellTypeNames.emplace_back(name);
    }
}

std::vector<std::string> const& GenomeMassEditService::getCellTypeNames() const
{
    return _cellTypeNames;
}

namespace
{
    template <typename T>
    bool isAllowed(std::optional<std::vector<T>> const& allowedValues, T const& value)
    {
        return !allowedValues || std::ranges::contains(*allowedValues, value);
    }

    bool fulfillsConditions(std::vector<JsonCondition> const& conditions, boost::json::value const& json)
    {
        return std::ranges::all_of(conditions, [&](JsonCondition const& condition) { return condition.isFulfilled(json); });
    }

    // The genes are moved out temporarily, so that serializing or patching the genome does not process them
    template <typename Func>
    void withoutGenes(GenomeDesc& genome, Func const& func)
    {
        auto genes = std::move(genome._genes);
        genome._genes.clear();
        func();
        genome._genes = std::move(genes);
    }

    template <typename Func>
    void withoutNodes(GeneDesc& gene, Func const& func)
    {
        auto nodes = std::move(gene._nodes);
        gene._nodes.clear();
        func();
        gene._nodes = std::move(nodes);
    }

    boost::json::value serializeGenomeHeader(GenomeDesc& genome)
    {
        boost::json::value result;
        withoutGenes(genome, [&] { result = SerializerService::get().serializeToJson(genome, false); });
        result.as_object().erase("genes");
        result.as_object().erase("id");
        return result;
    }

    void deserializeGenomeHeader(GenomeDesc& genome, boost::json::value const& json)
    {
        GenomeDesc header;
        SerializerService::get().deserializeFromJson(header, json);
        header._id = genome._id;
        header._genes = std::move(genome._genes);
        genome = std::move(header);
    }

    boost::json::value serializeGeneHeader(GeneDesc& gene)
    {
        boost::json::value result;
        withoutNodes(gene, [&] { result = SerializerService::get().serializeToJson(gene, false); });
        result.as_object().erase("nodes");
        return result;
    }

    void deserializeGeneHeader(GeneDesc& gene, boost::json::value const& json)
    {
        GeneDesc header;
        SerializerService::get().deserializeFromJson(header, json);
        header._nodes = std::move(gene._nodes);
        gene = std::move(header);
    }

    NeuralNetGenomeDesc const& getEmptyNeuralNetwork()
    {
        static NeuralNetGenomeDesc const result = [] {
            NeuralNetGenomeDesc neuralNetwork;
            neuralNetwork._weights.clear();
            neuralNetwork._biases.clear();
            neuralNetwork._activationFunctions.clear();
            neuralNetwork._connectionWeights.clear();
            return neuralNetwork;
        }();
        return result;
    }

    bool needsNeuralNetwork(JsonPath const& path)
    {
        auto const& segments = path.getSegments();
        if (segments.empty()) {
            return true;
        }
        auto const& segment = segments.front();
        return segment.type != JsonPath::Segment::Type::Key || segment.key == "neuralNetwork";
    }

    bool needsNeuralNetwork(std::vector<JsonCondition> const& conditions)
    {
        return std::ranges::any_of(conditions, [](JsonCondition const& condition) { return needsNeuralNetwork(condition.path); });
    }

    // The neural network makes up most of a node, so it is only serialized if needed
    boost::json::value serializeNode(NodeDesc const& node, bool withNeuralNetwork)
    {
        if (withNeuralNetwork) {
            return SerializerService::get().serializeToJson(node, false);
        }
        auto nodeWithoutNeuralNetwork = node;
        nodeWithoutNeuralNetwork._neuralNetwork = getEmptyNeuralNetwork();
        auto result = SerializerService::get().serializeToJson(nodeWithoutNeuralNetwork, false);
        result.as_object().erase("neuralNetwork");
        return result;
    }

    // Contains the subtree of the JSON at the leading field names of the path, so that patching with it only processes the changed part.
    // A patch merges objects instead of replacing them, so it is only suitable if the changed values are no objects or arrays.
    boost::json::value createPatch(boost::json::value const& json, JsonPath const& path)
    {
        std::vector<std::string> keys;
        for (auto const& segment : path.getSegments()) {
            if (segment.type != JsonPath::Segment::Type::Key) {
                break;
            }
            keys.emplace_back(segment.key);
        }
        auto subtree = &json;
        for (auto const& key : keys) {
            subtree = &subtree->as_object().at(key);
        }
        auto result = *subtree;
        for (auto const& key : keys | std::views::reverse) {
            result = boost::json::object{{key, std::move(result)}};
        }
        return result;
    }

    template <typename Desc>
    bool applyPatch(Desc& desc, boost::json::value const& patch)
    {
        auto origDesc = desc;
        SerializerService::get().deserializeFromJson(desc, patch, true);
        return desc != origDesc;
    }

    bool matchesNodeProperties(GenomeNodeFilter const& filter, NodeDesc const& node, int nodeIndex, std::vector<std::string> const& cellTypeNames)
    {
        if (!isAllowed(filter.indices, nodeIndex) || !isAllowed(filter.colors, node._color)) {
            return false;
        }
        if (filter.cellTypeNames && !std::ranges::contains(*filter.cellTypeNames, cellTypeNames.at(node._cellType.index()))) {
            return false;
        }
        if (filter.hasConstructor && *filter.hasConstructor != node._constructor.has_value()) {
            return false;
        }
        return filter.conditions.empty() || fulfillsConditions(filter.conditions, serializeNode(node, needsNeuralNetwork(filter.conditions)));
    }

    std::vector<int> getMatchingNodeIndices(GenomeNodeFilter const& filter, GeneDesc const& gene, std::vector<std::string> const& cellTypeNames)
    {
        std::vector<int> result;
        for (auto const& [nodeIndex, node] : gene._nodes | std::views::enumerate) {
            if (matchesNodeProperties(filter, node, toInt(nodeIndex), cellTypeNames)) {
                result.emplace_back(toInt(nodeIndex));
            }
        }
        return result;
    }

    bool matchesGene(GenomeGeneFilter const& filter, GeneDesc const& gene, int geneIndex, std::vector<std::string> const& cellTypeNames);

    bool hasMatchingNode(GenomeNodeFilter const& filter, GeneDesc const& gene, int geneIndex, std::vector<std::string> const& cellTypeNames)
    {
        if (!matchesGene(filter.gene, gene, geneIndex, cellTypeNames)) {
            return false;
        }
        return std::ranges::any_of(gene._nodes | std::views::enumerate, [&](auto const& indexAndNode) {
            auto const& [nodeIndex, node] = indexAndNode;
            return matchesNodeProperties(filter, node, toInt(nodeIndex), cellTypeNames);
        });
    }

    bool matchesGene(GenomeGeneFilter const& filter, GeneDesc const& gene, int geneIndex, std::vector<std::string> const& cellTypeNames)
    {
        if (!isAllowed(filter.indices, geneIndex) || !isAllowed(filter.names, gene._name)) {
            return false;
        }
        auto numNodes = toInt(gene._nodes.size());
        if ((filter.minNodes && numNodes < *filter.minNodes) || (filter.maxNodes && numNodes > *filter.maxNodes)) {
            return false;
        }
        if (filter.hasNode && !hasMatchingNode(*filter.hasNode, gene, geneIndex, cellTypeNames)) {
            return false;
        }
        if (!filter.conditions.empty()) {
            auto header = gene;
            return fulfillsConditions(filter.conditions, serializeGeneHeader(header));
        }
        return true;
    }

    std::vector<int> getMatchingGeneIndices(GenomeGeneFilter const& filter, GenomeDesc const& genome, std::vector<std::string> const& cellTypeNames)
    {
        std::vector<int> result;
        for (auto const& [geneIndex, gene] : genome._genes | std::views::enumerate) {
            if (matchesGene(filter, gene, toInt(geneIndex), cellTypeNames)) {
                result.emplace_back(toInt(geneIndex));
            }
        }
        return result;
    }

    int countNodeDifferences(std::vector<NodeDesc> const& nodes1, std::vector<NodeDesc> const& nodes2)
    {
        auto result = toInt(std::max(nodes1.size(), nodes2.size()) - std::min(nodes1.size(), nodes2.size()));
        for (auto const& [node1, node2] : std::views::zip(nodes1, nodes2)) {
            if (node1._cellType.index() != node2._cellType.index() || node1._constructor.has_value() != node2._constructor.has_value()) {
                ++result;
            }
        }
        return result;
    }

    // Number of node positions whose cell type or constructor presence differs or which exist in only one of the genomes
    int countNodeDifferences(GenomeDesc const& genome1, GenomeDesc const& genome2)
    {
        auto result = 0;
        for (auto const& [gene1, gene2] : std::views::zip(genome1._genes, genome2._genes)) {
            result += countNodeDifferences(gene1._nodes, gene2._nodes);
        }
        auto const& longerGenome = genome1._genes.size() > genome2._genes.size() ? genome1 : genome2;
        auto numCommonGenes = std::min(genome1._genes.size(), genome2._genes.size());
        for (auto const& gene : longerGenome._genes | std::views::drop(numCommonGenes)) {
            result += toInt(gene._nodes.size());
        }
        return result;
    }

    bool hasNodeColor(GenomeDesc const& genome, std::vector<int> const& colors)
    {
        return std::ranges::any_of(genome._genes, [&](GeneDesc const& gene) {
            return std::ranges::any_of(gene._nodes, [&](NodeDesc const& node) { return std::ranges::contains(colors, node._color); });
        });
    }

    bool matchesGenome(GenomeFilter const& filter, GenomeDesc const& genome, std::vector<std::string> const& cellTypeNames)
    {
        if (!isAllowed(filter.genomeIds, genome._id) || !isAllowed(filter.genomeNames, genome._name)) {
            return false;
        }
        if (filter.nodeColors && !hasNodeColor(genome, *filter.nodeColors)) {
            return false;
        }
        auto geneIndices = std::views::iota(0, toInt(genome._genes.size()));
        if (filter.hasNode && std::ranges::none_of(geneIndices, [&](int geneIndex) {
                return hasMatchingNode(*filter.hasNode, genome._genes.at(geneIndex), geneIndex, cellTypeNames);
            })) {
            return false;
        }
        if (filter.hasGene && std::ranges::none_of(geneIndices, [&](int geneIndex) {
                return matchesGene(*filter.hasGene, genome._genes.at(geneIndex), geneIndex, cellTypeNames);
            })) {
            return false;
        }
        if (filter.similarTo && countNodeDifferences(genome, *filter.similarTo) > filter.maxNodeDifferences) {
            return false;
        }
        if (!filter.conditions.empty()) {
            auto header = genome;
            return fulfillsConditions(filter.conditions, serializeGenomeHeader(header));
        }
        return true;
    }

    struct CreatureCells
    {
        int numCells = 0;
        uint32_t colorMask = 0;
        bool inArea = false;
    };

    uint32_t toColorBit(int color)
    {
        return color >= 0 && color < 32 ? 1u << color : 0u;
    }

    uint32_t toColorMask(std::vector<int> const& colors)
    {
        uint32_t result = 0;
        for (auto color : colors) {
            result |= toColorBit(color);
        }
        return result;
    }

    std::unordered_map<uint64_t, CreatureCells> collectCreatureCells(ContentDesc const& content, std::optional<RealRect> const& area)
    {
        std::unordered_map<uint64_t, CreatureCells> result;
        for (auto const& object : content._objects) {
            if (object.getObjectType() != ObjectType_Cell) {
                continue;
            }
            auto& cells = result[object.getCellRef()._creatureId];
            ++cells.numCells;
            cells.colorMask |= toColorBit(object._color);
            if (area && object._pos.x >= area->topLeft.x && object._pos.x <= area->bottomRight.x && object._pos.y >= area->topLeft.y
                && object._pos.y <= area->bottomRight.y) {
                cells.inArea = true;
            }
        }
        return result;
    }

    std::unordered_set<uint64_t> getCreatureIdsOfObjects(ContentDesc const& content, std::vector<uint64_t> const& objectIds)
    {
        std::unordered_set<uint64_t> remainingObjectIds(objectIds.begin(), objectIds.end());
        std::unordered_set<uint64_t> result;
        for (auto const& object : content._objects) {
            if (object.getObjectType() == ObjectType_Cell && remainingObjectIds.contains(object._id)) {
                result.insert(object.getCellRef()._creatureId);
            }
        }
        return result;
    }

    bool matchesCreature(
        GenomeFilter const& filter,
        CreatureDesc const& creature,
        CreatureCells const& cells,
        std::unordered_set<uint64_t> const& creatureIdsOfObjects)
    {
        if (!isAllowed(filter.creatureIds, creature._id) || !isAllowed(filter.lineageIds, creature._lineageId)
            || !isAllowed(filter.mutationStates, creature._mutationState)) {
            return false;
        }
        if (filter.objectIds && !creatureIdsOfObjects.contains(creature._id)) {
            return false;
        }
        if ((filter.minCells && cells.numCells < *filter.minCells) || (filter.maxCells && cells.numCells > *filter.maxCells)) {
            return false;
        }
        if ((filter.minGeneration && creature._generation < *filter.minGeneration) || (filter.maxGeneration && creature._generation > *filter.maxGeneration)) {
            return false;
        }
        if (filter.cellColors && (cells.colorMask & toColorMask(*filter.cellColors)) == 0) {
            return false;
        }
        return !filter.area || cells.inArea;
    }

    struct SelectedGenome
    {
        int genomeIndex = 0;
        std::vector<int> creatureIndices;
        bool sharedWithOtherCreatures = false;
    };

    std::vector<SelectedGenome> selectGenomes(ContentDesc const& content, GenomeFilter const& filter, std::vector<std::string> const& cellTypeNames)
    {
        std::unordered_map<uint64_t, int> genomeIndexById;
        for (auto const& [genomeIndex, genome] : content._genomes | std::views::enumerate) {
            genomeIndexById.emplace(genome._id, toInt(genomeIndex));
        }
        auto needsCells = filter.cellColors || filter.area || filter.minCells || filter.maxCells;
        auto creatureCells = needsCells ? collectCreatureCells(content, filter.area) : std::unordered_map<uint64_t, CreatureCells>();
        auto creatureIdsOfObjects = filter.objectIds ? getCreatureIdsOfObjects(content, *filter.objectIds) : std::unordered_set<uint64_t>();

        std::vector<std::optional<bool>> genomeMatches(content._genomes.size());
        std::vector<SelectedGenome> selectedGenomes(content._genomes.size());
        CreatureCells const noCells{};
        for (auto const& [creatureIndex, creature] : content._creatures | std::views::enumerate) {
            auto genomeIndexIter = genomeIndexById.find(creature._genomeId);
            if (genomeIndexIter == genomeIndexById.end()) {
                continue;
            }
            auto genomeIndex = genomeIndexIter->second;
            auto& selectedGenome = selectedGenomes.at(genomeIndex);
            auto cellsIter = creatureCells.find(creature._id);
            if (!matchesCreature(filter, creature, cellsIter != creatureCells.end() ? cellsIter->second : noCells, creatureIdsOfObjects)) {
                selectedGenome.sharedWithOtherCreatures = true;
                continue;
            }
            auto& genomeMatch = genomeMatches.at(genomeIndex);
            if (!genomeMatch) {
                genomeMatch = matchesGenome(filter, content._genomes.at(genomeIndex), cellTypeNames);
            }
            if (*genomeMatch) {
                selectedGenome.creatureIndices.emplace_back(toInt(creatureIndex));
            }
        }

        std::vector<SelectedGenome> result;
        for (auto&& [genomeIndex, selectedGenome] : selectedGenomes | std::views::enumerate) {
            if (!selectedGenome.creatureIndices.empty()) {
                selectedGenome.genomeIndex = toInt(genomeIndex);
                result.emplace_back(std::move(selectedGenome));
            }
        }
        return result;
    }

    template <typename Operation>
    concept ValueOperation = requires(Operation const& operation) {
        { operation.path } -> std::convertible_to<JsonPath const&>;
    };

    struct RandomDraws
    {
        std::optional<double> genome;
        std::optional<double> gene;
        std::optional<double> node;
    };

    double drawUniform()
    {
        return std::clamp(NumberGenerator::get().getRandomDouble(), 0.0, 1.0);
    }

    double drawNormal()
    {
        auto u1 = std::max(drawUniform(), 1e-12);
        auto u2 = drawUniform();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
    }

    template <typename Draw>
    double getDraw(GenomeEditGranularity granularity, RandomDraws& draws, Draw const& draw)
    {
        auto getCachedDraw = [&](std::optional<double>& cachedDraw) {
            if (!cachedDraw) {
                cachedDraw = draw();
            }
            return *cachedDraw;
        };
        switch (granularity) {
        case GenomeEditGranularity::Genome:
            return getCachedDraw(draws.genome);
        case GenomeEditGranularity::Gene:
            return getCachedDraw(draws.gene);
        case GenomeEditGranularity::Node:
            return getCachedDraw(draws.node);
        default:
            return draw();
        }
    }

    bool isInteger(boost::json::value const& value)
    {
        return value.is_int64() || value.is_uint64();
    }

    boost::json::value toNumber(boost::json::value const& original, double value)
    {
        if (isInteger(original)) {
            return static_cast<int64_t>(std::llround(value));
        }
        return value;
    }

    std::optional<boost::json::value> transformValue(GenomeSetOperation const& operation, boost::json::value const&, RandomDraws&)
    {
        return operation.value;
    }

    std::optional<boost::json::value> transformValue(GenomeAddOperation const& operation, boost::json::value const& value, RandomDraws&)
    {
        if (!value.is_number()) {
            return std::nullopt;
        }
        return toNumber(value, value.to_number<double>() + operation.summand);
    }

    std::optional<boost::json::value> transformValue(GenomeScaleOperation const& operation, boost::json::value const& value, RandomDraws&)
    {
        if (!value.is_number()) {
            return std::nullopt;
        }
        return toNumber(value, value.to_number<double>() * operation.factor);
    }

    std::optional<boost::json::value> transformValue(GenomeClampOperation const& operation, boost::json::value const& value, RandomDraws&)
    {
        if (!value.is_number()) {
            return std::nullopt;
        }
        auto result = value.to_number<double>();
        if (operation.min) {
            result = std::max(result, isInteger(value) ? std::ceil(*operation.min) : *operation.min);
        }
        if (operation.max) {
            result = std::min(result, isInteger(value) ? std::floor(*operation.max) : *operation.max);
        }
        return toNumber(value, result);
    }

    std::optional<boost::json::value> transformValue(GenomeRandomizeOperation const& operation, boost::json::value const& value, RandomDraws& draws)
    {
        auto draw = getDraw(operation.granularity, draws, drawUniform);
        if (!operation.values.empty()) {
            auto numValues = toInt(operation.values.size());
            return operation.values.at(std::min(toInt(draw * numValues), numValues - 1));
        }
        if (!value.is_number()) {
            return std::nullopt;
        }
        auto min = operation.min.value_or(0);
        auto max = operation.max.value_or(min);
        if (isInteger(value) && std::ceil(min) <= std::floor(max)) {
            auto lowest = std::ceil(min);
            auto highest = std::floor(max);
            return toNumber(value, std::min(lowest + std::floor(draw * (highest - lowest + 1)), highest));
        }
        return toNumber(value, min + draw * (max - min));
    }

    std::optional<boost::json::value> transformValue(GenomeJitterOperation const& operation, boost::json::value const& value, RandomDraws& draws)
    {
        if (!value.is_number()) {
            return std::nullopt;
        }
        return toNumber(value, value.to_number<double>() + operation.sigma * getDraw(operation.granularity, draws, drawNormal));
    }

    std::optional<boost::json::value> transformValue(GenomeRemapOperation const& operation, boost::json::value const& value, RandomDraws&)
    {
        for (auto const& entry : operation.entries) {
            if (JsonCondition::areEqual(value, entry.from)) {
                return entry.to;
            }
        }
        return value;
    }

    bool isStructured(boost::json::value const& value)
    {
        return value.is_object() || value.is_array();
    }

    template <typename Operation>
    bool assignsStructuredValues(Operation const& operation)
    {
        if constexpr (std::is_same_v<Operation, GenomeSetOperation>) {
            return isStructured(operation.value);
        } else if constexpr (std::is_same_v<Operation, GenomeRemapOperation>) {
            return std::ranges::any_of(operation.entries, [](GenomeRemapEntry const& entry) { return isStructured(entry.to); });
        } else if constexpr (std::is_same_v<Operation, GenomeRandomizeOperation>) {
            return std::ranges::any_of(operation.values, isStructured);
        } else {
            return false;
        }
    }

    template <typename Operation>
    bool applyToValues(Operation const& operation, boost::json::value& json, RandomDraws& draws, GenomeEditStatistics& statistics)
    {
        auto result = false;
        for (auto value : operation.path.find(json)) {
            ++statistics.numValues;
            auto newValue = transformValue(operation, *value, draws);
            if (!newValue) {
                ++statistics.numNonNumericValues;
            } else if (!JsonCondition::areEqual(*value, *newValue)) {
                *value = std::move(*newValue);
                ++statistics.numChangedValues;
                result = true;
            }
        }
        return result;
    }

    void applyToGenome(GenomeEditOperation const& operation, GenomeDesc& genome, RandomDraws& draws, GenomeEditStatistics& statistics)
    {
        ++statistics.numTargets;
        std::visit(
            [&]<typename Operation>(Operation const& concreteOperation) {
                if constexpr (std::is_same_v<Operation, GenomePatchOperation>) {
                    ++statistics.numValues;
                    withoutGenes(genome, [&] {
                        if (applyPatch(genome, concreteOperation.patch)) {
                            ++statistics.numChangedValues;
                        }
                    });
                } else if constexpr (std::is_same_v<Operation, GenomeInsertGeneOperation>) {
                    auto numGenes = toInt(genome._genes.size());
                    auto geneIndex = std::min(concreteOperation.index.value_or(numGenes), numGenes);
                    GenomeDescEditService::get().addGene(genome, geneIndex - 1, concreteOperation.gene);
                } else if constexpr (ValueOperation<Operation>) {
                    auto json = serializeGenomeHeader(genome);
                    if (!applyToValues(concreteOperation, json, draws, statistics)) {
                        return;
                    }
                    if (assignsStructuredValues(concreteOperation)) {
                        deserializeGenomeHeader(genome, json);
                    } else {
                        withoutGenes(genome, [&] { SerializerService::get().deserializeFromJson(genome, createPatch(json, concreteOperation.path), true); });
                    }
                }
            },
            operation);
    }

    void applyToGene(GenomeEditOperation const& operation, GeneDesc& gene, RandomDraws& draws, GenomeEditStatistics& statistics)
    {
        std::visit(
            [&]<typename Operation>(Operation const& concreteOperation) {
                if constexpr (std::is_same_v<Operation, GenomePatchOperation>) {
                    ++statistics.numValues;
                    withoutNodes(gene, [&] {
                        if (applyPatch(gene, concreteOperation.patch)) {
                            ++statistics.numChangedValues;
                        }
                    });
                } else if constexpr (std::is_same_v<Operation, GenomeInsertNodeOperation>) {
                    auto numNodes = toInt(gene._nodes.size());
                    auto nodeIndex = std::min(concreteOperation.index.value_or(numNodes), numNodes);
                    gene._nodes.insert(gene._nodes.begin() + nodeIndex, concreteOperation.node);
                } else if constexpr (std::is_same_v<Operation, GenomeReplaceGeneOperation>) {
                    gene = concreteOperation.gene;
                } else if constexpr (ValueOperation<Operation>) {
                    auto json = serializeGeneHeader(gene);
                    if (!applyToValues(concreteOperation, json, draws, statistics)) {
                        return;
                    }
                    if (assignsStructuredValues(concreteOperation)) {
                        deserializeGeneHeader(gene, json);
                    } else {
                        withoutNodes(gene, [&] { SerializerService::get().deserializeFromJson(gene, createPatch(json, concreteOperation.path), true); });
                    }
                }
            },
            operation);
    }

    void applyToNode(GenomeEditOperation const& operation, NodeDesc& node, RandomDraws& draws, GenomeEditStatistics& statistics)
    {
        std::visit(
            [&]<typename Operation>(Operation const& concreteOperation) {
                if constexpr (std::is_same_v<Operation, GenomePatchOperation>) {
                    ++statistics.numValues;
                    if (applyPatch(node, concreteOperation.patch)) {
                        ++statistics.numChangedValues;
                    }
                } else if constexpr (ValueOperation<Operation>) {
                    auto replacesObjects = assignsStructuredValues(concreteOperation);
                    auto json = serializeNode(node, replacesObjects || needsNeuralNetwork(concreteOperation.path));
                    if (!applyToValues(concreteOperation, json, draws, statistics)) {
                        return;
                    }
                    if (replacesObjects) {
                        SerializerService::get().deserializeFromJson(node, json);
                    } else {
                        SerializerService::get().deserializeFromJson(node, createPatch(json, concreteOperation.path), true);
                    }
                }
            },
            operation);
    }

    void applyToGenes(
        GenomeEdit const& edit,
        GenomeDesc& genome,
        RandomDraws& draws,
        GenomeEditStatistics& statistics,
        std::vector<std::string> const& cellTypeNames)
    {
        auto geneIndices = getMatchingGeneIndices(edit.geneFilter, genome, cellTypeNames);
        statistics.numTargets += toInt(geneIndices.size());
        if (std::holds_alternative<GenomeRemoveGenesOperation>(edit.operation)) {
            GenomeDescEditService::get().removeGenes(genome, std::set(geneIndices.begin(), geneIndices.end()));
            return;
        }
        for (auto geneIndex : geneIndices) {
            draws.gene.reset();
            applyToGene(edit.operation, genome._genes.at(geneIndex), draws, statistics);
        }
    }

    void applyToNodes(
        GenomeEdit const& edit,
        GenomeDesc& genome,
        RandomDraws& draws,
        GenomeEditStatistics& statistics,
        std::vector<std::string> const& cellTypeNames)
    {
        for (auto geneIndex : getMatchingGeneIndices(edit.nodeFilter.gene, genome, cellTypeNames)) {
            auto& gene = genome._genes.at(geneIndex);
            auto nodeIndices = getMatchingNodeIndices(edit.nodeFilter, gene, cellTypeNames);
            statistics.numTargets += toInt(nodeIndices.size());
            if (std::holds_alternative<GenomeRemoveNodesOperation>(edit.operation)) {
                for (auto nodeIndex : nodeIndices | std::views::reverse) {
                    GenomeDescEditService::get().removeNode(gene, nodeIndex);
                }
                continue;
            }
            draws.gene.reset();
            for (auto nodeIndex : nodeIndices) {
                draws.node.reset();
                applyToNode(edit.operation, gene._nodes.at(nodeIndex), draws, statistics);
            }
        }
    }

    void applyEdit(GenomeEdit const& edit, GenomeDesc& genome, GenomeEditStatistics& statistics, std::vector<std::string> const& cellTypeNames)
    {
        auto origNumTargets = statistics.numTargets;
        RandomDraws draws;
        switch (edit.level) {
        case GenomeEditLevel::Genome:
            applyToGenome(edit.operation, genome, draws, statistics);
            break;
        case GenomeEditLevel::Gene:
            applyToGenes(edit, genome, draws, statistics, cellTypeNames);
            break;
        case GenomeEditLevel::Node:
            applyToNodes(edit, genome, draws, statistics, cellTypeNames);
            break;
        }
        if (statistics.numTargets > origNumTargets) {
            ++statistics.numGenomesWithTargets;
        }
    }

    void collectChanges(boost::json::value const& before, boost::json::value const& after, std::string const& path, GenomeChanges& result)
    {
        auto addChange = [&](GenomeChange&& change) {
            ++result.numChanges;
            if (result.firstChanges.size() < MaxExampleChanges) {
                result.firstChanges.emplace_back(std::move(change));
            }
        };
        auto getChildPath = [&](boost::json::string_view key) { return path.empty() ? std::string(key) : path + "." + std::string(key); };

        if (before.is_object() && after.is_object()) {
            for (auto const& [key, valueAfter] : after.as_object()) {
                if (auto valueBefore = before.as_object().if_contains(key)) {
                    collectChanges(*valueBefore, valueAfter, getChildPath(key), result);
                } else {
                    addChange({.path = getChildPath(key), .after = valueAfter});
                }
            }
            for (auto const& [key, valueBefore] : before.as_object()) {
                if (!after.as_object().contains(key)) {
                    addChange({.path = getChildPath(key), .before = valueBefore});
                }
            }
        } else if (before.is_array() && after.is_array()) {
            auto const& arrayBefore = before.as_array();
            auto const& arrayAfter = after.as_array();
            if (arrayBefore.size() != arrayAfter.size()) {
                addChange({.path = path, .arraySizes = std::make_pair(toInt(arrayBefore.size()), toInt(arrayAfter.size()))});
                return;
            }
            for (auto const& [index, elementBefore, elementAfter] : std::views::zip(std::views::iota(0), arrayBefore, arrayAfter)) {
                collectChanges(elementBefore, elementAfter, std::format("{}[{}]", path, index), result);
            }
        } else if (!JsonCondition::areEqual(before, after)) {
            addChange({.path = path, .before = before, .after = after});
        }
    }

    GenomeChanges describeChanges(GenomeDesc const& before, GenomeDesc const& after)
    {
        auto jsonBefore = SerializerService::get().serializeToJson(before, false);
        auto jsonAfter = SerializerService::get().serializeToJson(after, false);
        jsonBefore.as_object().erase("id");
        jsonAfter.as_object().erase("id");
        GenomeChanges result;
        collectChanges(jsonBefore, jsonAfter, "", result);
        return result;
    }

    std::string getIssueTitle(GenomeIssueType type)
    {
        switch (type) {
        case GenomeIssueType::VoidBoundaryNode:
            return "void node at gene boundary";
        case GenomeIssueType::NodesCutOffFromLastNode:
            return "nodes cut off from the last node";
        case GenomeIssueType::CycleAvoidingRootGene:
            return "cycle avoiding the root gene";
        case GenomeIssueType::GeneUnreachableFromRoot:
            return "gene unreachable from the root gene";
        case GenomeIssueType::TooManyGenesWithSeparation:
            return "too many genes with separation";
        case GenomeIssueType::ConstructsRemovedGene:
            return "constructs a removed gene";
        case GenomeIssueType::InjectsRemovedGene:
            return "injects a removed gene";
        }
        return "unknown issue";
    }

    std::optional<std::string> findNewIssue(GenomeDesc const& before, GenomeDesc const& after)
    {
        auto getPrimaryIssues = [](GenomeDesc const& genome) {
            auto issues = DescValidationService::get().findGenomeIssues(genome);
            std::erase_if(issues, [](GenomeIssue const& issue) { return issue.isFollowUp(); });
            return issues;
        };
        auto issuesBefore = getPrimaryIssues(before);
        auto issuesAfter = getPrimaryIssues(after);
        if (issuesAfter.size() <= issuesBefore.size()) {
            return std::nullopt;
        }
        auto isSameIssue = [](GenomeIssue const& issue1, GenomeIssue const& issue2) {
            return issue1.type == issue2.type && issue1.geneIndex == issue2.geneIndex && issue1.nodeIndex == issue2.nodeIndex;
        };
        auto newIssue = std::ranges::find_if(issuesAfter, [&](GenomeIssue const& issue) {
            return std::ranges::none_of(issuesBefore, [&](GenomeIssue const& issueBefore) { return isSameIssue(issue, issueBefore); });
        });
        auto const& issue = newIssue != issuesAfter.end() ? *newIssue : issuesAfter.back();
        auto location = issue.nodeIndex ? std::format("gene {}, node {}", issue.geneIndex, *issue.nodeIndex) : std::format("gene {}", issue.geneIndex);
        return std::format("{}: {}", location, getIssueTitle(issue.type));
    }

    std::optional<uint64_t> findCellOfCreature(ContentDesc const& content, uint64_t creatureId)
    {
        for (auto const& object : content._objects) {
            if (object.getObjectType() == ObjectType_Cell && object.getCellRef()._creatureId == creatureId) {
                return object._id;
            }
        }
        return std::nullopt;
    }

    uint64_t getMaxEntityId(ContentDesc const& content)
    {
        uint64_t result = 0;
        for (auto const& object : content._objects) {
            result = std::max(result, object._id);
        }
        for (auto const& energy : content._energies) {
            result = std::max(result, energy._id);
        }
        for (auto const& creature : content._creatures) {
            result = std::max(result, creature._id);
        }
        for (auto const& genome : content._genomes) {
            result = std::max(result, genome._id);
        }
        return result;
    }
}

GenomeEditResult GenomeMassEditService::edit(ContentDesc& content, GenomeFilter const& filter, std::vector<GenomeEdit> const& edits) const
{
    checkFilter(filter);
    for (auto const& [editIndex, genomeEdit] : edits | std::views::enumerate) {
        try {
            checkEdit(genomeEdit);
        } catch (std::invalid_argument const& exception) {
            throw std::invalid_argument(std::format("edits[{}]: {}", editIndex, exception.what()));
        }
    }

    GenomeEditResult result;
    result.editStatistics.resize(edits.size());
    auto selectedGenomes = selectGenomes(content, filter, _cellTypeNames);
    result.numGenomes = toInt(selectedGenomes.size());

    std::vector<std::pair<SelectedGenome const*, GenomeDesc>> modifiedGenomes;
    for (auto const& selectedGenome : selectedGenomes) {
        result.numCreatures += toInt(selectedGenome.creatureIndices.size());

        auto const& origGenome = content._genomes.at(selectedGenome.genomeIndex);
        auto genome = origGenome;
        for (auto&& [editIndex, genomeEdit, statistics] : std::views::zip(std::views::iota(0), edits, result.editStatistics)) {
            try {
                applyEdit(genomeEdit, genome, statistics, _cellTypeNames);
            } catch (std::invalid_argument const& exception) {
                throw std::invalid_argument(std::format("edits[{}] on genome {}: {}", editIndex, origGenome._id, exception.what()));
            }
        }
        if (genome == origGenome) {
            continue;
        }

        auto editedGenome = genome;
        DescValidationService::get().validateAndCorrect(genome);
        if (genome != editedGenome) {
            if (result.numCorrectedGenomes == 0) {
                result.exampleCorrections = describeChanges(editedGenome, genome);
            }
            ++result.numCorrectedGenomes;
        }
        if (genome == origGenome) {
            continue;
        }
        if (genome._genes.empty() && !origGenome._genes.empty()) {
            throw std::invalid_argument(std::format("The edits would remove all genes of genome {}. Nothing was changed.", origGenome._id));
        }

        ++result.numModifiedGenomes;
        if (selectedGenome.sharedWithOtherCreatures) {
            ++result.numCopiedGenomes;
        }
        if (auto newIssue = findNewIssue(origGenome, genome)) {
            if (!result.exampleIssue) {
                result.exampleIssue = newIssue;
            }
            ++result.numGenomesWithNewIssues;
        }
        if (!result.exampleGenomeId) {
            result.exampleGenomeId = origGenome._id;
            result.exampleCellId = findCellOfCreature(content, content._creatures.at(selectedGenome.creatureIndices.front())._id);
            result.exampleChanges = describeChanges(origGenome, genome);
        }
        modifiedGenomes.emplace_back(&selectedGenome, std::move(genome));
    }

    NumberGenerator::get().adaptMaxEntityId(getMaxEntityId(content));
    for (auto& [selectedGenome, genome] : modifiedGenomes) {
        if (!selectedGenome->sharedWithOtherCreatures) {
            content._genomes.at(selectedGenome->genomeIndex) = std::move(genome);
            continue;
        }
        genome._id = NumberGenerator::get().createEntityId();
        if (selectedGenome == modifiedGenomes.front().first) {
            result.exampleGenomeId = genome._id;
        }
        for (auto creatureIndex : selectedGenome->creatureIndices) {
            content._creatures.at(creatureIndex)._genomeId = genome._id;
        }
        content._genomes.emplace_back(std::move(genome));
    }
    return result;
}

namespace
{
    GenomeCountStatistics calcCountStatistics(std::vector<int> const& counts, std::vector<int> const& weights)
    {
        if (counts.empty()) {
            return {};
        }
        auto [min, max] = std::ranges::minmax(counts);
        auto sum = 0.0;
        auto sumOfWeights = 0.0;
        for (auto const& [count, weight] : std::views::zip(counts, weights)) {
            sum += count * weight;
            sumOfWeights += weight;
        }
        return {.min = min, .max = max, .mean = sum / sumOfWeights};
    }

    template <typename Key>
    std::vector<std::pair<Key, int>> sortByFrequency(std::map<Key, int> const& frequencies)
    {
        std::vector<std::pair<Key, int>> result(frequencies.begin(), frequencies.end());
        std::ranges::stable_sort(result, [](auto const& entry1, auto const& entry2) { return entry1.second > entry2.second; });
        return result;
    }

    class FieldStatisticsAccumulator
    {
    public:
        void add(boost::json::value const& value, int weight)
        {
            _result.numValues += weight;
            if (value.is_number()) {
                auto number = value.to_number<double>();
                _result.min = _result.numNumbers == 0 ? number : std::min(_result.min, number);
                _result.max = _result.numNumbers == 0 ? number : std::max(_result.max, number);
                _sum += number * weight;
                _result.numNumbers += weight;
            }
            if (_hasManyDistinctValues) {
                return;
            }
            auto key = boost::json::serialize(value);
            auto [iter, inserted] = _frequencies.try_emplace(key, value, 0);
            iter->second.second += weight;
            if (_frequencies.size() > MaxDistinctValues) {
                _hasManyDistinctValues = true;
                _frequencies.clear();
            }
        }

        GenomeFieldStatistics getResult() const
        {
            auto result = _result;
            if (result.numNumbers > 0) {
                result.mean = _sum / toDouble(result.numNumbers);
            }
            std::vector<std::pair<boost::json::value, int>> frequentValues;
            for (auto const& [key, valueAndCount] : _frequencies) {
                frequentValues.emplace_back(valueAndCount);
            }
            std::ranges::stable_sort(frequentValues, [](auto const& entry1, auto const& entry2) { return entry1.second > entry2.second; });
            if (frequentValues.size() > MaxFrequentValues) {
                frequentValues.resize(MaxFrequentValues);
            }
            result.frequentValues = std::move(frequentValues);
            return result;
        }

    private:
        GenomeFieldStatistics _result;
        double _sum = 0;
        std::map<std::string, std::pair<boost::json::value, int>> _frequencies;
        bool _hasManyDistinctValues = false;
    };

    void collectFieldValues(
        GenomeFieldQuery const& field,
        GenomeDesc const& genome,
        int weight,
        FieldStatisticsAccumulator& accumulator,
        std::vector<std::string> const& cellTypeNames)
    {
        auto addValues = [&](boost::json::value const& json) {
            for (auto value : field.path.find(json)) {
                accumulator.add(*value, weight);
            }
        };
        switch (field.level) {
        case GenomeEditLevel::Genome: {
            auto header = genome;
            addValues(serializeGenomeHeader(header));
            break;
        }
        case GenomeEditLevel::Gene:
            for (auto geneIndex : getMatchingGeneIndices(field.geneFilter, genome, cellTypeNames)) {
                auto header = genome._genes.at(geneIndex);
                addValues(serializeGeneHeader(header));
            }
            break;
        case GenomeEditLevel::Node:
            for (auto geneIndex : getMatchingGeneIndices(field.nodeFilter.gene, genome, cellTypeNames)) {
                auto const& gene = genome._genes.at(geneIndex);
                for (auto nodeIndex : getMatchingNodeIndices(field.nodeFilter, gene, cellTypeNames)) {
                    addValues(serializeNode(gene._nodes.at(nodeIndex), needsNeuralNetwork(field.path)));
                }
            }
            break;
        }
    }

    std::vector<uint64_t> findCellsOfCreatures(ContentDesc const& content, std::vector<uint64_t> const& creatureIds)
    {
        std::unordered_map<uint64_t, uint64_t> cellIdByCreatureId;
        std::unordered_set<uint64_t> remainingCreatureIds(creatureIds.begin(), creatureIds.end());
        for (auto const& object : content._objects) {
            if (object.getObjectType() != ObjectType_Cell) {
                continue;
            }
            auto creatureId = object.getCellRef()._creatureId;
            if (remainingCreatureIds.erase(creatureId) > 0) {
                cellIdByCreatureId.emplace(creatureId, object._id);
            }
        }
        std::vector<uint64_t> result;
        for (auto creatureId : creatureIds) {
            if (auto cellId = cellIdByCreatureId.find(creatureId); cellId != cellIdByCreatureId.end()) {
                result.emplace_back(cellId->second);
            }
        }
        return result;
    }
}

GenomeSummary GenomeMassEditService::summarize(
    ContentDesc const& content,
    GenomeFilter const& filter,
    std::vector<GenomeFieldQuery> const& fields,
    bool countPerCreature) const
{
    checkFilter(filter);
    for (auto const& [fieldIndex, field] : fields | std::views::enumerate) {
        try {
            checkFieldQuery(field);
        } catch (std::invalid_argument const& exception) {
            throw std::invalid_argument(std::format("fields[{}]: {}", fieldIndex, exception.what()));
        }
    }

    auto selectedGenomes = selectGenomes(content, filter, _cellTypeNames);

    GenomeSummary result;
    result.numGenomes = toInt(selectedGenomes.size());
    std::map<int, int> numCreaturesByLineage;
    std::map<std::string, int> numNodesByCellType;
    std::map<int, int> numNodesByColor;
    std::vector<int> numGenesPerGenome;
    std::vector<int> numNodesPerGenome;
    std::vector<int> weights;
    std::vector<FieldStatisticsAccumulator> accumulators(fields.size());
    std::vector<uint64_t> exampleCreatureIds;
    for (auto const& selectedGenome : selectedGenomes) {
        auto numCreatures = toInt(selectedGenome.creatureIndices.size());
        result.numCreatures += numCreatures;
        for (auto creatureIndex : selectedGenome.creatureIndices) {
            ++numCreaturesByLineage[content._creatures.at(creatureIndex)._lineageId];
        }
        if (exampleCreatureIds.size() < MaxExampleObjects) {
            exampleCreatureIds.emplace_back(content._creatures.at(selectedGenome.creatureIndices.front())._id);
        }

        auto weight = countPerCreature ? numCreatures : 1;
        auto const& genome = content._genomes.at(selectedGenome.genomeIndex);
        auto numNodes = 0;
        for (auto const& gene : genome._genes) {
            for (auto const& node : gene._nodes) {
                numNodesByCellType[_cellTypeNames.at(node._cellType.index())] += weight;
                numNodesByColor[node._color] += weight;
                ++numNodes;
            }
        }
        numGenesPerGenome.emplace_back(toInt(genome._genes.size()));
        numNodesPerGenome.emplace_back(numNodes);
        weights.emplace_back(weight);

        for (auto&& [field, accumulator] : std::views::zip(fields, accumulators)) {
            collectFieldValues(field, genome, weight, accumulator, _cellTypeNames);
        }
    }

    result.numLineages = toInt(numCreaturesByLineage.size());
    result.largestLineages = sortByFrequency(numCreaturesByLineage);
    if (result.largestLineages.size() > MaxLargestLineages) {
        result.largestLineages.resize(MaxLargestLineages);
    }
    result.genesPerGenome = calcCountStatistics(numGenesPerGenome, weights);
    result.nodesPerGenome = calcCountStatistics(numNodesPerGenome, weights);
    result.numNodesByCellType = sortByFrequency(numNodesByCellType);
    result.numNodesByColor = sortByFrequency(numNodesByColor);
    for (auto const& accumulator : accumulators) {
        result.fields.emplace_back(accumulator.getResult());
    }
    result.exampleCellIds = findCellsOfCreatures(content, exampleCreatureIds);
    return result;
}

boost::json::value const& GenomeMassEditService::getFormat(GenomeEditLevel level) const
{
    switch (level) {
    case GenomeEditLevel::Gene:
        return _geneFormat;
    case GenomeEditLevel::Node:
        return _nodeFormat;
    default:
        return _genomeFormat;
    }
}

namespace
{
    void checkCondition(JsonCondition const& condition, boost::json::value const& format)
    {
        condition.path.resolveFormat(format);
        auto const& pathText = condition.path.getText();
        switch (condition.op) {
        case JsonConditionOperator::Less:
        case JsonConditionOperator::LessEqual:
        case JsonConditionOperator::Greater:
        case JsonConditionOperator::GreaterEqual:
            if (!condition.value.is_number()) {
                throw std::invalid_argument(std::format("The comparison of '{}' needs a number.", pathText));
            }
            break;
        case JsonConditionOperator::In:
            if (!condition.value.is_array()) {
                throw std::invalid_argument(std::format("The condition 'in' on '{}' needs an array of values.", pathText));
            }
            break;
        case JsonConditionOperator::Exists:
            if (!condition.value.is_bool()) {
                throw std::invalid_argument(std::format("The condition 'exists' on '{}' needs a boolean.", pathText));
            }
            break;
        default:
            break;
        }
    }

    void checkRange(std::optional<int> const& min, std::optional<int> const& max, std::string const& name)
    {
        if (min && max && *min > *max) {
            throw std::invalid_argument(std::format("The minimum {} must not be greater than the maximum.", name));
        }
    }
}

void GenomeMassEditService::checkFilter(GenomeFilter const& filter) const
{
    checkRange(filter.minCells, filter.maxCells, "number of cells");
    checkRange(filter.minGeneration, filter.maxGeneration, "generation");
    if (filter.maxNodeDifferences < 0) {
        throw std::invalid_argument("The maximum number of node differences must not be negative.");
    }
    if (filter.hasNode) {
        checkNodeFilter(*filter.hasNode);
    }
    if (filter.hasGene) {
        checkGeneFilter(*filter.hasGene);
    }
    for (auto const& condition : filter.conditions) {
        checkCondition(condition, _genomeFormat);
    }
}

void GenomeMassEditService::checkGeneFilter(GenomeGeneFilter const& filter) const
{
    checkRange(filter.minNodes, filter.maxNodes, "number of nodes");
    if (filter.hasNode) {
        checkNodeFilter(*filter.hasNode);
    }
    for (auto const& condition : filter.conditions) {
        checkCondition(condition, _geneFormat);
    }
}

void GenomeMassEditService::checkNodeFilter(GenomeNodeFilter const& filter) const
{
    checkGeneFilter(filter.gene);
    if (filter.cellTypeNames) {
        for (auto const& cellType : *filter.cellTypeNames) {
            if (!std::ranges::contains(_cellTypeNames, cellType)) {
                throw std::invalid_argument(std::format("Unknown cell type '{}', allowed are: {}.", cellType, StringHelper::formatEnumeration(_cellTypeNames)));
            }
        }
    }
    for (auto const& condition : filter.conditions) {
        checkCondition(condition, _nodeFormat);
    }
}

namespace
{
    std::string getLevelName(GenomeEditLevel level)
    {
        switch (level) {
        case GenomeEditLevel::Gene:
            return "gene";
        case GenomeEditLevel::Node:
            return "node";
        default:
            return "genome";
        }
    }

    void checkLevel(GenomeEditLevel level, GenomeEditLevel requiredLevel, std::string const& operationName)
    {
        if (level != requiredLevel) {
            throw std::invalid_argument(std::format("{} needs the level '{}'.", operationName, getLevelName(requiredLevel)));
        }
    }

    void checkGranularity(GenomeEditGranularity granularity, GenomeEditLevel level)
    {
        auto isValid = [&] {
            switch (granularity) {
            case GenomeEditGranularity::Node:
                return level == GenomeEditLevel::Node;
            case GenomeEditGranularity::Gene:
                return level != GenomeEditLevel::Genome;
            default:
                return true;
            }
        }();
        if (!isValid) {
            throw std::invalid_argument(std::format(
                "Random values cannot be drawn per {} on the level '{}'.", granularity == GenomeEditGranularity::Node ? "node" : "gene", getLevelName(level)));
        }
    }

    bool isCompatible(boost::json::value const& format, boost::json::value const& value)
    {
        if (format.is_null() || value.is_null()) {
            return true;
        }
        if (format.is_number()) {
            return value.is_number();
        }
        if (format.is_bool()) {
            return value.is_bool();
        }
        if (format.is_string()) {
            return value.is_string();
        }
        if (format.is_array()) {
            return value.is_array();
        }
        return value.is_object();
    }

    void checkValue(boost::json::value const& value, std::vector<boost::json::value const*> const& formats, JsonPath const& path)
    {
        if (std::ranges::none_of(formats, [&](auto const* format) { return isCompatible(*format, value); })) {
            throw std::invalid_argument(std::format("The value {} does not match the type of '{}'.", boost::json::serialize(value), path.getText()));
        }
    }

    void checkLeadsToNumbers(std::vector<boost::json::value const*> const& formats, JsonPath const& path)
    {
        if (std::ranges::none_of(formats, [](auto const* format) { return format->is_number() || format->is_null(); })) {
            throw std::invalid_argument(std::format("'{}' does not lead to numbers.", path.getText()));
        }
    }

    void checkPatch(boost::json::value const& patch, GenomeEditLevel level)
    {
        if (!patch.is_object()) {
            throw std::invalid_argument("A patch must be a JSON object.");
        }
        auto const& patchObject = patch.as_object();
        if (level == GenomeEditLevel::Genome && patchObject.contains("genes")) {
            throw std::invalid_argument("Genes cannot be patched on the level 'genome', use the levels 'gene' or 'node'.");
        }
        if (level == GenomeEditLevel::Genome && patchObject.contains("id")) {
            throw std::invalid_argument("The genome id cannot be changed.");
        }
        if (level == GenomeEditLevel::Gene && patchObject.contains("nodes")) {
            throw std::invalid_argument("Nodes cannot be patched on the level 'gene', use the level 'node'.");
        }
        switch (level) {
        case GenomeEditLevel::Genome: {
            GenomeDesc genome;
            SerializerService::get().deserializeFromJson(genome, patch, true);
            break;
        }
        case GenomeEditLevel::Gene: {
            GeneDesc gene;
            SerializerService::get().deserializeFromJson(gene, patch, true);
            break;
        }
        case GenomeEditLevel::Node: {
            NodeDesc node;
            SerializerService::get().deserializeFromJson(node, patch, true);
            break;
        }
        }
    }
}

void GenomeMassEditService::checkEdit(GenomeEdit const& edit) const
{
    if (edit.level == GenomeEditLevel::Gene) {
        checkGeneFilter(edit.geneFilter);
    }
    if (edit.level == GenomeEditLevel::Node) {
        checkNodeFilter(edit.nodeFilter);
    }
    std::visit(
        [&]<typename Operation>(Operation const& operation) {
            if constexpr (std::is_same_v<Operation, GenomePatchOperation>) {
                checkPatch(operation.patch, edit.level);
            } else if constexpr (std::is_same_v<Operation, GenomeInsertNodeOperation>) {
                checkLevel(edit.level, GenomeEditLevel::Gene, "Inserting nodes");
                if (operation.index && *operation.index < 0) {
                    throw std::invalid_argument("The index of an inserted node must not be negative.");
                }
            } else if constexpr (std::is_same_v<Operation, GenomeRemoveNodesOperation>) {
                checkLevel(edit.level, GenomeEditLevel::Node, "Removing nodes");
            } else if constexpr (std::is_same_v<Operation, GenomeInsertGeneOperation>) {
                checkLevel(edit.level, GenomeEditLevel::Genome, "Inserting genes");
                if (operation.index && *operation.index < 1) {
                    throw std::invalid_argument("Gene 0 is the root gene, insert genes at index 1 or later.");
                }
            } else if constexpr (std::is_same_v<Operation, GenomeReplaceGeneOperation>) {
                checkLevel(edit.level, GenomeEditLevel::Gene, "Replacing genes");
            } else if constexpr (std::is_same_v<Operation, GenomeRemoveGenesOperation>) {
                checkLevel(edit.level, GenomeEditLevel::Gene, "Removing genes");
            } else if constexpr (ValueOperation<Operation>) {
                auto formats = operation.path.resolveFormat(getFormat(edit.level));
                if constexpr (std::is_same_v<Operation, GenomeSetOperation>) {
                    checkValue(operation.value, formats, operation.path);
                } else if constexpr (std::is_same_v<Operation, GenomeRemapOperation>) {
                    if (operation.entries.empty()) {
                        throw std::invalid_argument("The remapping needs at least one entry.");
                    }
                    for (auto const& entry : operation.entries) {
                        checkValue(entry.to, formats, operation.path);
                    }
                } else if constexpr (std::is_same_v<Operation, GenomeRandomizeOperation>) {
                    checkGranularity(operation.granularity, edit.level);
                    if (!operation.values.empty()) {
                        for (auto const& value : operation.values) {
                            checkValue(value, formats, operation.path);
                        }
                    } else {
                        checkLeadsToNumbers(formats, operation.path);
                        if (!operation.min || !operation.max) {
                            throw std::invalid_argument("Randomizing needs either values or a minimum and a maximum.");
                        }
                        if (*operation.min > *operation.max) {
                            throw std::invalid_argument("The minimum must not be greater than the maximum.");
                        }
                    }
                } else {
                    checkLeadsToNumbers(formats, operation.path);
                    if constexpr (std::is_same_v<Operation, GenomeJitterOperation>) {
                        checkGranularity(operation.granularity, edit.level);
                        if (operation.sigma < 0) {
                            throw std::invalid_argument("The standard deviation must not be negative.");
                        }
                    } else if constexpr (std::is_same_v<Operation, GenomeClampOperation>) {
                        if (!operation.min && !operation.max) {
                            throw std::invalid_argument("Clamping needs a minimum or a maximum.");
                        }
                        if (operation.min && operation.max && *operation.min > *operation.max) {
                            throw std::invalid_argument("The minimum must not be greater than the maximum.");
                        }
                    }
                }
            }
        },
        edit.operation);
}

void GenomeMassEditService::checkFieldQuery(GenomeFieldQuery const& field) const
{
    if (field.level == GenomeEditLevel::Gene) {
        checkGeneFilter(field.geneFilter);
    }
    if (field.level == GenomeEditLevel::Node) {
        checkNodeFilter(field.nodeFilter);
    }
    field.path.resolveFormat(getFormat(field.level));
}
