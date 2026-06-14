#include "GenomeDescEditService.h"

#include <algorithm>
#include <deque>
#include <iterator>
#include <map>
#include <set>

#include <boost/range/adaptors.hpp>

#include <EngineInterface/NumberGenerator.h>

#include "DescEditService.h"
#include "GenomeDescInfoService.h"

namespace
{
    auto constexpr PreviewColor = 0;
}

void GenomeDescEditService::addGene(GenomeDesc& genome, int index, GeneDesc const& newGene) const
{
    if (genome._genes.empty()) {
        genome._genes.emplace_back(newGene);
        return;
    }

    for (int i = 0; i < genome._genes.size(); ++i) {
        auto& gene = genome._genes[i];
        for (auto& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto& constructor = node._constructor.value();
                if (constructor._geneIndex > index) {
                    ++constructor._geneIndex;
                }
            }
        }
    }

    genome._genes.insert(genome._genes.begin() + index + 1, newGene);
}

void GenomeDescEditService::removeGene(GenomeDesc& genome, int index) const
{
    for (int i = 0; i < genome._genes.size(); ++i) {
        if (i == index) {
            continue;
        }
        auto& gene = genome._genes[i];
        for (auto& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto& constructor = node._constructor.value();
                if (constructor._geneIndex >= index) {
                    --constructor._geneIndex;
                }
            }
        }
    }
    genome._genes.erase(genome._genes.begin() + index);
}

void GenomeDescEditService::swapGenes(GenomeDesc& genome, int index) const
{
    std::swap(genome._genes.at(index), genome._genes.at(index + 1));

    for (auto& gene : genome._genes) {
        for (auto& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto& constructor = node._constructor.value();
                if (constructor._geneIndex == index) {
                    constructor._geneIndex = index + 1;
                } else if (constructor._geneIndex == index + 1) {
                    constructor._geneIndex = index;
                }
            }
        }
    }
}

void GenomeDescEditService::addNode(GeneDesc& gene, int index, NodeDesc const& node) const
{
    if (gene._nodes.empty()) {
        gene._nodes.emplace_back(node);
        return;
    }

    gene._nodes.insert(gene._nodes.begin() + index + 1, node);
}

void GenomeDescEditService::removeNode(GeneDesc& gene, int index) const
{
    auto deleteAtStart = index == 0;
    auto deleteAtEnd = index == gene._nodes.size() - 1;
    gene._nodes.erase(gene._nodes.begin() + index);
    if (!gene._nodes.empty()) {
        if (deleteAtStart) {
            gene._nodes.front()._referenceAngle = 0;
        }
        if (deleteAtEnd) {
            gene._nodes.back()._referenceAngle = 0;
        }
    }
}

void GenomeDescEditService::swapNodes(GeneDesc& gene, int index) const
{
    std::swap(gene._nodes.at(index), gene._nodes.at(index + 1));
}

namespace
{
    struct GeneNodeInfo
    {
        int geneIndex;
        int depth;  // Distance from start gene (for BFS ordering)
        int parentGeneIndex = -1;
        int parentNodeIndex = -1;
        int numBranches = 1;
        int numConcatenations = 1;
    };

    int multiplyAndClamp(int left, int right)
    {
        if (left == std::numeric_limits<int>::max() || right == std::numeric_limits<int>::max()) {
            return std::numeric_limits<int>::max();
        }
        auto result = static_cast<int64_t>(left) * right;
        return result > std::numeric_limits<int>::max() ? std::numeric_limits<int>::max() : toInt(result);
    }

    // Collects all genes in the reference tree using breadth-first search
    std::vector<GeneNodeInfo> collectGenesInBFS(GenomeDesc const& genome, int startGeneIndex)
    {
        std::vector<GeneNodeInfo> result;
        std::set<int> visited;
        std::deque<GeneNodeInfo> queue;

        queue.push_back({startGeneIndex, 0});
        visited.insert(startGeneIndex);

        while (!queue.empty()) {
            auto current = queue.front();
            queue.pop_front();
            result.push_back(current);

            if (current.geneIndex >= genome._genes.size()) {
                continue;
            }

            auto const& gene = genome._genes.at(current.geneIndex);
            for (auto const& [nodeIndex, node] : gene._nodes | boost::adaptors::indexed(0)) {
                if (node._constructor.has_value()) {
                    auto const& constructor = node._constructor.value();
                    if (constructor._geneIndex < genome._genes.size() && !visited.contains(constructor._geneIndex)) {
                        auto const effectiveNumBranches = constructor._separation ? 1 : constructor._numBranches;
                        queue.push_back(
                            {constructor._geneIndex,
                             current.depth + 1,
                             current.geneIndex,
                             toInt(nodeIndex),
                             multiplyAndClamp(current.numBranches, effectiveNumBranches),
                             multiplyAndClamp(current.numConcatenations, constructor._numConcatenations)});
                        visited.insert(constructor._geneIndex);
                    }
                }
            }
        }

        return result;
    }

    // Returns true if genome has been trimmed
    bool trimNodes(GenomeDesc& genome, int& nodeCounter, int startGeneIndex, int nodeLimit)
    {
        // Collect all genes in breadth-first order
        auto genesInBFS = collectGenesInBFS(genome, startGeneIndex);

        // Calculate total nodes needed (with bounded concatenations)
        int64_t totalNodesNeeded = 0;
        for (auto const& geneInfo : genesInBFS) {
            if (geneInfo.geneIndex >= genome._genes.size()) {
                continue;
            }
            auto const& gene = genome._genes.at(geneInfo.geneIndex);
            auto truncatedNumConcatenations = std::min(1000000, geneInfo.numConcatenations);  // Prevent overflow
            totalNodesNeeded += static_cast<int64_t>(gene._nodes.size()) * geneInfo.numBranches * truncatedNumConcatenations;
        }

        // If we're under the limit, no trimming needed
        if (totalNodesNeeded <= nodeLimit) {
            nodeCounter = static_cast<int>(totalNodesNeeded);
            return false;
        }

        // We need to trim - distribute budget proportionally across all genes
        bool trimmed = false;
        int remainingBudget = nodeLimit;

        // Allocate budget to each gene proportionally, ensuring genes at each depth get adequate representation
        std::map<int, int> geneNodeBudget;

        // First, ensure every gene gets at least one full copy of its nodes if possible
        for (auto const& geneInfo : genesInBFS) {
            if (geneInfo.geneIndex >= genome._genes.size()) {
                continue;
            }

            auto& gene = genome._genes.at(geneInfo.geneIndex);
            if (gene._nodes.empty()) {
                geneNodeBudget[geneInfo.geneIndex] = 0;
                continue;
            }

            int nodesInGene = toInt(gene._nodes.size());
            if (remainingBudget >= nodesInGene) {
                geneNodeBudget[geneInfo.geneIndex] = nodesInGene;
                remainingBudget -= nodesInGene;
            } else {
                geneNodeBudget[geneInfo.geneIndex] = remainingBudget;
                remainingBudget = 0;
            }
        }

        // Then, distribute remaining budget proportionally
        if (remainingBudget > 0) {
            for (auto const& geneInfo : genesInBFS) {
                if (geneInfo.geneIndex >= genome._genes.size()) {
                    continue;
                }

                auto& gene = genome._genes.at(geneInfo.geneIndex);
                if (gene._nodes.empty()) {
                    continue;
                }

                // Calculate additional budget proportionally
                auto truncatedNumConcatenations = std::min(1000000, geneInfo.numConcatenations);
                int64_t idealNodes = static_cast<int64_t>(gene._nodes.size()) * geneInfo.numBranches * truncatedNumConcatenations;
                int64_t additionalBudget = (idealNodes * remainingBudget) / std::max(static_cast<int64_t>(1), totalNodesNeeded);

                geneNodeBudget[geneInfo.geneIndex] += static_cast<int>(additionalBudget);
            }
        }

        // Apply the budget to each gene
        for (auto const& geneInfo : genesInBFS) {
            if (geneInfo.geneIndex >= genome._genes.size()) {
                continue;
            }

            auto& gene = genome._genes.at(geneInfo.geneIndex);
            int budget = geneNodeBudget[geneInfo.geneIndex];

            if (budget == 0) {
                // No budget for this gene - remove all nodes
                gene._nodes.clear();
                trimmed = true;
                continue;
            }

            int nodesPerCopy = toInt(gene._nodes.size());
            if (nodesPerCopy == 0) {
                continue;
            }

            // Calculate how many concatenations we can afford
            auto nodesPerConcatenation = nodesPerCopy * geneInfo.numBranches;
            auto affordableConcatenations = budget / nodesPerConcatenation;

            // If we can't afford even one full copy, trim nodes
            if (affordableConcatenations == 0) {
                gene._nodes.resize(budget / geneInfo.numBranches);
                if (geneInfo.parentGeneIndex != -1) {
                    genome._genes.at(geneInfo.parentGeneIndex)._nodes.at(geneInfo.parentNodeIndex)._constructor->_numConcatenations = 1;
                }
                trimmed = true;

                // Castrate constructors in trimmed nodes
                for (auto& node : gene._nodes) {
                    if (node._constructor.has_value()) {
                        auto& constructor = node._constructor.value();
                        constructor._geneIndex = toInt(genome._genes.size());
                    }
                }
            } else if (
                geneInfo.parentGeneIndex != -1
                && affordableConcatenations < genome._genes.at(geneInfo.parentGeneIndex)._nodes.at(geneInfo.parentNodeIndex)._constructor->_numConcatenations) {
                // Reduce concatenations to fit budget
                genome._genes.at(geneInfo.parentGeneIndex)._nodes.at(geneInfo.parentNodeIndex)._constructor->_numConcatenations = affordableConcatenations;
                trimmed = true;
            }

            auto actualNumConcatenations = geneInfo.parentGeneIndex == -1
                ? geneInfo.numConcatenations
                : genome._genes.at(geneInfo.parentGeneIndex)._nodes.at(geneInfo.parentNodeIndex)._constructor->_numConcatenations;
            nodeCounter += toInt(gene._nodes.size()) * geneInfo.numBranches * actualNumConcatenations;
        }

        return trimmed;
    }
}

std::vector<SubGenomeDesc> GenomeDescEditService::createSubGenomesForPreview(
    GenomeDesc const& genome,
    std::vector<GeneIndicesForSubGenome> const& geneIndicesForSubGenomes,
    bool detailSimulation) const
{
    std::vector<SubGenomeDesc> result;
    for (auto const& geneIndicesForSubGenome : geneIndicesForSubGenomes) {
        auto subGenome = genome;
        adaptDescriptionForPreview(subGenome, geneIndicesForSubGenome, detailSimulation);
        result.emplace_back(subGenome, geneIndicesForSubGenome.front());
    }

    // Trim sub-genomes if too many cells (use simple heuristics)
    int sumNumResultingCells = 0;
    for (auto const& subGenomeWithStartGeneIndex : result) {
        auto subGenome = subGenomeWithStartGeneIndex.genome;
        auto startGeneIndex = subGenomeWithStartGeneIndex.startIndex;

        auto resultingCells = GenomeDescInfoService::get().getNumberOfResultingCells(subGenome, startGeneIndex);
        if (resultingCells != -1) {
            sumNumResultingCells += resultingCells;
        } else {
            // Infinite number of cells => force trimming
            sumNumResultingCells = PREVIEW_MAX_CELLS + 1;
        }
    }
    if (sumNumResultingCells > PREVIEW_MAX_CELLS) {

        for (int i = 0, numSubGenomes = toInt(result.size()); i < numSubGenomes; ++i) {
            auto& subGenome = result.at(i).genome;
            auto startGeneIndex = result.at(i).startIndex;

            int nodeCounter = 0;
            auto trimmed = trimNodes(subGenome, nodeCounter, startGeneIndex, PREVIEW_MAX_CELLS / numSubGenomes);
            if (trimmed) {
                result.at(i).trimmed = true;
            }
        }
    }

    return result;
}

auto GenomeDescEditService::createSeedCollectionForPreview(
    std::vector<SubGenomeDesc> const& subGenomes,
    std::optional<std::reference_wrapper<GenotypeToPhenotypeCache const>> cache) const -> SeedCollectionResult
{
    auto const& editService = DescEditService::get();

    RealVector2D currentPos{toFloat(PREVIEW_HEIGHT) / 2, toFloat(PREVIEW_HEIGHT) / 2};

    SeedCollectionResult result;

    for (auto const& subGenome : subGenomes) {
        std::optional<Desc> cachedValue;

        // Try to get from cache if provided
        if (cache.has_value()) {
            cachedValue = cache.value().get().find(subGenome);
        }

        if (cachedValue.has_value()) {
            auto cachedPhenotype = cachedValue.value();
            editService.setCenter(cachedPhenotype, currentPos);

            CHECK(cachedPhenotype._creatures.size() <= 2);
            auto seedFirst = false;
            if (cachedPhenotype._creatures.front()._generation == 0) {
                seedFirst = true;  // first Creature is seed
            }

            result.description.add(std::move(cachedPhenotype), false);  // Try keeping ids stable for preview selection

            auto index = seedFirst ? result.description._creatures.size() - cachedPhenotype._creatures.size()
                                   : result.description._creatures.size() - cachedPhenotype._creatures.size() + 1;
            result.seedCreatureIds.emplace_back(result.description._creatures.at(index)._id);
        } else {
            auto seed = createSeedForPreview(subGenome, currentPos);
            result.description.add(std::move(seed), true);

            result.seedCreatureIds.emplace_back(result.description._creatures.back()._id);
        }
        currentPos.x += toFloat(PREVIEW_HEIGHT) / 2;  // Adjust position for the next sub-genome
    }
    return result;
}

std::vector<Desc> GenomeDescEditService::extractPhenotypesFromPreview(Desc&& preview, std::vector<uint64_t> const& seedCreatureIds) const
{
    std::unordered_map<uint64_t, int> creatureIdToIndex;
    for (auto const& [index, creatureId] : seedCreatureIds | boost::adaptors::indexed(0)) {
        creatureIdToIndex.insert_or_assign(creatureId, toInt(index));
    }
    auto cache = preview.createCache();
    std::vector<Desc> result(seedCreatureIds.size());
    for (auto& creature : preview._creatures) {
        if (creature._generation == 0) {
            auto genomeIndex = cache->genomeIdToIndex.at(creature._genomeId);
            auto const& genome = preview._genomes.at(genomeIndex);

            auto index = creatureIdToIndex.at(creature._id);
            result.at(index)._creatures.emplace_back(std::move(creature));
            result.at(index)._genomes.emplace_back(genome);
        } else {
            CHECK(creature._generation == 1);

            auto index = creatureIdToIndex.at(creature._ancestorId.value());
            result.at(index)._creatures.emplace_back(std::move(creature));

            // Genome already added from the seed creature (should be the same since no mutations in preview)
        }
    }
    for (auto& object : preview._objects) {
        auto creatureIndex = cache->creatureIdToIndex.at(object.getCellRef()._creatureId);
        auto& creature = preview._creatures.at(creatureIndex);
        auto phenotypeIndex = creatureIdToIndex.at(creature._generation == 0 ? creature._id : creature._ancestorId.value());
        result.at(phenotypeIndex)._objects.emplace_back(std::move(object));
    }
    return result;
}

void GenomeDescEditService::removeSeedFromPhenotype(Desc& phenotype) const
{
    std::set<uint64_t> seedCellIds;
    std::map<uint64_t, uint64_t> creatureIdToIndex;
    for (auto const& [creatureIndex, creature] : phenotype._creatures | boost::adaptors::indexed(0)) {
        creatureIdToIndex.emplace(creature._id, toInt(creatureIndex));
    }
    for (auto const& object : phenotype._objects) {
        auto const& creature = phenotype._creatures.at(creatureIdToIndex.at(object.getCellRef()._creatureId));
        if (creature._generation == 0) {
            seedCellIds.insert(object._id);
        }
    }
    DescEditService::get().removeCellIf(phenotype, [&seedCellIds](auto const& object) { return seedCellIds.contains(object._id); });
}

Desc GenomeDescEditService::createSeedForPreview(SubGenomeDesc const& subGenome, RealVector2D const& pos) const
{
    Desc result;
    result._genomes.emplace_back(subGenome.genome);
    auto creature = CreatureDesc().genomeId(subGenome.genome._id);
    result._creatures.emplace_back(creature);
    result._objects.emplace_back(ObjectDesc()
                                     .color(PreviewColor)
                                     .stiffness(1.0f)
                                     .pos(pos)
                                     .type(CellDesc()
                                               .headCell(true)
                                               .creatureId(creature._id)
                                               .constructor(ConstructorDesc()
                                                                .autoTriggerInterval(100)
                                                                .provideEnergy(ProvideEnergy_Free)
                                                                .geneIndex(subGenome.startIndex)
                                                                .separation(true)
                                                                .numBranches(1)
                                                                .numConcatenations(1))));
    return result;
}

namespace
{
    void castrate(GenomeDesc& genome, int geneIndex, std::set<int>& inspectedGeneIndices)
    {
        if (geneIndex >= genome._genes.size() || inspectedGeneIndices.contains(geneIndex)) {
            return;
        }
        inspectedGeneIndices.insert(geneIndex);
        auto& gene = genome._genes.at(geneIndex);
        for (auto& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto& constructor = node._constructor.value();
                if (constructor._geneIndex < genome._genes.size()) {
                    if (inspectedGeneIndices.contains(constructor._geneIndex)) {
                        constructor._geneIndex = genome._genes.size();  // Recursive part => perform castration
                    } else {
                        castrate(genome, constructor._geneIndex, inspectedGeneIndices);  // Inspect further gene

                        if (constructor._separation || constructor._geneIndex == 0) {
                            // A separating reference or a reference to the root gene starts a new creature (see ConstructorProcessor),
                            // => perform castration.
                            constructor._geneIndex = genome._genes.size();
                        }
                    }
                }
            }
        }
        inspectedGeneIndices.erase(geneIndex);
    }

    void adaptGenomeAttributesForPreview(GenomeDesc& genome, bool detailSimulation)
    {
        genome._lineageId = 0;
        genome._accumulatedMutations = 0;
        genome._mutationRates._neuronMutations[0] = NeuronMutationDesc();
        genome._mutationRates._neuronMutations[1] = NeuronMutationDesc();
        genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc();
        genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc();
        genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc();
        genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc();
        for (auto& gene : genome._genes) {
            gene._homogeneCellType = false;
            for (auto& node : gene._nodes) {
                node._color = PreviewColor;
                if (!detailSimulation) {
                    node._neuralNetwork = NeuralNetGenomeDesc();
                    if (node.getCellType() != CellType_Void) {
                        node._cellType = BaseGenomeDesc();
                    }
                }
                if (node._constructor.has_value()) {
                    auto& constructor = node._constructor.value();
                    constructor._autoTriggerInterval = 50;
                    constructor._constructionActivationTime = 10;
                    constructor._reservedEnergy = 0;
                }
            }
        }
    }

    void resetNames(GenomeDesc& genome)
    {
        genome._name.clear();
        for (auto& gene : genome._genes) {
            gene._name.clear();
        }
    }

    void resetUnusedGenes(GenomeDesc& genome, GeneIndicesForSubGenome const& geneIndices)
    {
        std::set<int> geneIndexSet(geneIndices.begin(), geneIndices.end());
        for (int i = 0, size = toInt(genome._genes.size()); i < size; ++i) {
            if (!geneIndexSet.contains(i)) {
                genome._genes.at(i) = GeneDesc();
            }
        }
    }
}

void GenomeDescEditService::adaptDescriptionForPreview(GenomeDesc& genome, GeneIndicesForSubGenome const& geneIndices, bool detailSimulation) const
{
    auto startGeneIndex = geneIndices.front();

    std::set<int> inspectedGeneIndices;
    castrate(genome, startGeneIndex, inspectedGeneIndices);
    adaptGenomeAttributesForPreview(genome, detailSimulation);
    resetNames(genome);
    if (!detailSimulation) {
        genome._frontAngle = 0;
    }

    resetUnusedGenes(genome, geneIndices);
}
