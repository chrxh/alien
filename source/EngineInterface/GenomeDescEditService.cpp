#include "GenomeDescEditService.h"

#include <algorithm>
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
    void castrateConstructor(GenomeDesc& genome, ConstructorGenomeDesc& constructor)
    {
        constructor._geneIndex = toInt(genome._genes.size());
    }

    int getEffectiveNumBranches(ConstructorGenomeDesc const& constructor)
    {
        return std::max(1, constructor._separation ? 1 : constructor._numBranches);
    }

    // Trims the nodes of the gene and of all genes referenced by it such that a single instance of the gene results in at most nodeLimit cells
    // when every constructor builds its referenced gene exactly once. Returns the number of cells resulting from one instance of the gene.
    int trimNodesOfGene(GenomeDesc& genome, int geneIndex, int nodeLimit, std::set<int>& activeGeneIndices, bool& trimmed)
    {
        if (geneIndex >= toInt(genome._genes.size())) {
            return 0;
        }

        auto& gene = genome._genes.at(geneIndex);
        if (toInt(gene._nodes.size()) > nodeLimit) {
            gene._nodes.resize(nodeLimit);
            trimmed = true;

            // No budget left for referenced genes
            for (auto& node : gene._nodes) {
                if (node._constructor.has_value()) {
                    castrateConstructor(genome, node._constructor.value());
                }
            }
            return nodeLimit;
        }

        activeGeneIndices.insert(geneIndex);

        std::vector<int> constructorNodeIndices;
        for (auto const& [nodeIndex, node] : gene._nodes | boost::adaptors::indexed(0)) {
            if (node._constructor.has_value()) {
                constructorNodeIndices.emplace_back(toInt(nodeIndex));
            }
        }

        auto numCells = toInt(gene._nodes.size());
        auto remainingBudget = nodeLimit - numCells;
        auto numRemainingConstructors = toInt(constructorNodeIndices.size());

        for (auto const& nodeIndex : constructorNodeIndices) {
            auto budgetShare = remainingBudget / numRemainingConstructors;
            --numRemainingConstructors;

            auto& constructor = genome._genes.at(geneIndex)._nodes.at(nodeIndex)._constructor.value();
            auto referencedGeneIndex = constructor._geneIndex;
            if (referencedGeneIndex >= toInt(genome._genes.size())) {
                continue;
            }
            if (activeGeneIndices.contains(referencedGeneIndex)) {
                castrateConstructor(genome, constructor);  // Recursive reference => perform castration
                trimmed = true;
                continue;
            }

            auto numBranches = getEffectiveNumBranches(constructor);
            auto budgetPerBranch = budgetShare / numBranches;
            if (budgetPerBranch == 0) {
                castrateConstructor(genome, constructor);  // No budget left => perform castration
                trimmed = true;
                continue;
            }

            // The referenced gene is built once per branch, i.e. its cells count multiple times
            auto numCellsForReference = numBranches * trimNodesOfGene(genome, referencedGeneIndex, budgetPerBranch, activeGeneIndices, trimmed);
            numCells += numCellsForReference;
            remainingBudget -= numCellsForReference;
        }

        activeGeneIndices.erase(geneIndex);
        return numCells;
    }

    // Returns the number of cells resulting from one instance of the gene if all constructors are limited to maxNumConcatenations concatenations.
    // The result is clamped to upperBound + 1.
    int64_t getNumCells(GenomeDesc const& genome, int geneIndex, int maxNumConcatenations, int64_t upperBound, std::set<int>& activeGeneIndices)
    {
        if (geneIndex >= toInt(genome._genes.size()) || activeGeneIndices.contains(geneIndex)) {
            return 0;
        }
        activeGeneIndices.insert(geneIndex);

        auto const& gene = genome._genes.at(geneIndex);
        int64_t result = toInt(gene._nodes.size());
        for (auto const& node : gene._nodes) {
            if (result > upperBound) {
                break;
            }
            if (node._constructor.has_value()) {
                auto const& constructor = node._constructor.value();
                auto numConcatenations = std::min(constructor._numConcatenations, maxNumConcatenations);
                result += static_cast<int64_t>(getEffectiveNumBranches(constructor)) * numConcatenations
                    * getNumCells(genome, constructor._geneIndex, maxNumConcatenations, upperBound, activeGeneIndices);
            }
        }

        activeGeneIndices.erase(geneIndex);
        return std::min(result, upperBound + 1);
    }

    // Returns true if concatenations have been reduced
    bool applyMaxNumConcatenations(GenomeDesc& genome, int geneIndex, int maxNumConcatenations, std::set<int>& visitedGeneIndices)
    {
        if (geneIndex >= toInt(genome._genes.size()) || visitedGeneIndices.contains(geneIndex)) {
            return false;
        }
        visitedGeneIndices.insert(geneIndex);

        auto result = false;
        for (auto& node : genome._genes.at(geneIndex)._nodes) {
            if (node._constructor.has_value()) {
                auto& constructor = node._constructor.value();
                if (constructor._geneIndex >= toInt(genome._genes.size())) {
                    continue;
                }
                if (constructor._numConcatenations > maxNumConcatenations) {
                    constructor._numConcatenations = maxNumConcatenations;
                    result = true;
                }
                result |= applyMaxNumConcatenations(genome, constructor._geneIndex, maxNumConcatenations, visitedGeneIndices);
            }
        }
        return result;
    }

    // Returns true if genome has been trimmed
    bool trimNodes(GenomeDesc& genome, int startGeneIndex, int nodeLimit)
    {
        bool trimmed = false;
        std::set<int> activeGeneIndices;
        trimNodesOfGene(genome, startGeneIndex, nodeLimit, activeGeneIndices, trimmed);

        // Search for the largest number of concatenations that all constructors can use uniformly.
        // Concatenations multiply along the reference chain, thus a uniform limit distributes the cells evenly over all nesting levels.
        auto lowerLimit = 1;
        auto upperLimit = std::max(1, nodeLimit);
        while (lowerLimit < upperLimit) {
            auto middleLimit = lowerLimit + (upperLimit - lowerLimit + 1) / 2;
            activeGeneIndices.clear();
            if (getNumCells(genome, startGeneIndex, middleLimit, nodeLimit, activeGeneIndices) <= nodeLimit) {
                lowerLimit = middleLimit;
            } else {
                upperLimit = middleLimit - 1;
            }
        }

        std::set<int> visitedGeneIndices;
        trimmed |= applyMaxNumConcatenations(genome, startGeneIndex, lowerLimit, visitedGeneIndices);

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
        auto numSubGenomes = toInt(result.size());
        for (auto& subGenome : result) {
            subGenome.trimmed = trimNodes(subGenome.genome, subGenome.startIndex, PREVIEW_MAX_CELLS / numSubGenomes);
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
        std::optional<ContentDesc> cachedValue;

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
                seedFirst = true;  // First Creature is seed
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

std::vector<ContentDesc> GenomeDescEditService::extractPhenotypesFromPreview(ContentDesc&& preview, std::vector<uint64_t> const& seedCreatureIds) const
{
    std::unordered_map<uint64_t, int> creatureIdToIndex;
    for (auto const& [index, creatureId] : seedCreatureIds | boost::adaptors::indexed(0)) {
        creatureIdToIndex.insert_or_assign(creatureId, toInt(index));
    }
    auto cache = preview.createCache();
    std::vector<ContentDesc> result(seedCreatureIds.size());
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

void GenomeDescEditService::removeSeedFromPhenotype(ContentDesc& phenotype) const
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

ContentDesc GenomeDescEditService::createSeedForPreview(SubGenomeDesc const& subGenome, RealVector2D const& pos) const
{
    ContentDesc result;
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
        genome._mutationRates._neuronMutations[0] = NeuronMutationDesc();
        genome._mutationRates._neuronMutations[1] = NeuronMutationDesc();
        genome._mutationRates._connectionMutations[0] = ConnectionMutationDesc();
        genome._mutationRates._connectionMutations[1] = ConnectionMutationDesc();
        genome._mutationRates._cellTypePropertiesMutations[0] = CellTypePropertiesMutationDesc();
        genome._mutationRates._cellTypePropertiesMutations[1] = CellTypePropertiesMutationDesc();
        genome._mutationRates._geometryMutations[0] = GeometryMutationDesc();
        genome._mutationRates._geometryMutations[1] = GeometryMutationDesc();
        for (auto& gene : genome._genes) {
            if (!detailSimulation) {
                gene._homogeneousCellType = false;
            }
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
