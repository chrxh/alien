#include "GenomeDescriptionEditService.h"

#include <algorithm>
#include <iterator>

#include <boost/range/adaptors.hpp>

#include <EngineInterface/NumberGenerator.h>

#include "DescriptionEditService.h"
#include "GenomeDescriptionInfoService.h"

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
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDesc>(node._cellType);
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
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDesc>(node._cellType);
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
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDesc>(node._cellType);
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
    gene._nodes.erase(gene._nodes.begin() + index);
}

void GenomeDescEditService::swapNodes(GeneDesc& gene, int index) const
{
    std::swap(gene._nodes.at(index), gene._nodes.at(index + 1));
}

namespace
{
    // Returns true if genome has been trimmed
    bool trimNodes(GenomeDesc& genome, int& nodeCounter, int startGeneIndex, int nodeLimit)
    {
        auto result = false;
        auto& gene = genome._genes.at(startGeneIndex);

        // Trim nodes if limit is exceeded
        if (nodeCounter + toInt(gene._nodes.size()) > nodeLimit) {
            auto newSize = std::max(0, nodeLimit - nodeCounter);
            gene._nodes.resize(newSize);
            gene._numConcatenations = 1;
            for (auto& node : gene._nodes) {
                if (node.getCellType() == CellTypeGenome_Constructor) {
                    auto& constructor = std::get<ConstructorGenomeDesc>(node._cellType);
                    constructor._geneIndex = toInt(genome._genes.size());  // Castrate further construction
                }
            }
            return true;
        }

        // Trim concatenations if limit is exceeded
        auto truncatedNumConcatenations = std::min(1000000, gene._numConcatenations);  // Prevent overflow
        if (nodeCounter + gene._nodes.size() * truncatedNumConcatenations > nodeLimit) {
            gene._numConcatenations = (nodeLimit - nodeCounter) / toInt(gene._nodes.size());
            result = true;
        }

        nodeCounter += toInt(gene._nodes.size()) * gene._numConcatenations;

        // Continue with constructor nodes
        for (auto const& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto const& constructor = std::get<ConstructorGenomeDesc>(node._cellType);
                if (constructor._geneIndex < genome._genes.size()) {
                    result |= trimNodes(genome, nodeCounter, constructor._geneIndex, nodeLimit);
                }
            }
        }

        return result;
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
    auto const& editService = DescriptionEditService::get();

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
    for (auto& object: preview._objects) {
        auto creatureIndex = cache->creatureIdToIndex.at(*object.getCellRef()._creatureId);
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
        if (object.getCellRef()._creatureId.has_value()) {
            auto const& creature = phenotype._creatures.at(creatureIdToIndex.at(object.getCellRef()._creatureId.value()));
            if (creature._generation == 0) {
                seedCellIds.insert(object._id);
            }
        }
    }
    DescriptionEditService::get().removeCellIf(phenotype, [&seedCellIds](auto const& object) { return seedCellIds.contains(object._id); });
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
                                   .type(CellDesc().creatureId(creature._id).cellType(ConstructorDesc().provideEnergy(ProvideEnergy_FreeGeneration).geneIndex(subGenome.startIndex))));
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
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDesc>(node._cellType);
                if (constructor._geneIndex < genome._genes.size()) {
                    if (inspectedGeneIndices.contains(constructor._geneIndex)) {
                        constructor._geneIndex = genome._genes.size();  // Recursive part => perform castration
                    } else {
                        castrate(genome, constructor._geneIndex, inspectedGeneIndices);  // Inspect further gene

                        auto& refGene = genome._genes.at(constructor._geneIndex);
                        if (refGene._separation) {
                            constructor._geneIndex = genome._genes.size();  // Separated part => perform castration
                        }
                    }
                }
            }
        }
        inspectedGeneIndices.erase(geneIndex);
    }

    void adaptNodeAttributesForPreview(GenomeDesc& genome, bool detailSimulation)
    {
        for (auto& gene : genome._genes) {
            for (auto& node : gene._nodes) {
                node._color = PreviewColor;
                if (!detailSimulation) {
                    node._neuralNetwork = NeuralNetworkGenomeDesc();
                    node._signalRestriction = SignalRestrictionGenomeDesc();
                }
                if (node.getCellType() != CellTypeGenome_Constructor) {
                    if (!detailSimulation) {
                        node._cellType = BaseGenomeDesc();
                    }
                } else {
                    auto& constructor = std::get<ConstructorGenomeDesc>(node._cellType);
                    constructor._autoTriggerInterval = 75;
                    constructor._constructionActivationTime = 200;
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

void GenomeDescEditService::adaptDescriptionForPreview(GenomeDesc& genome, GeneIndicesForSubGenome const& geneIndices, bool detailSimulation)
    const
{
    auto startGeneIndex = geneIndices.front();

    std::set<int> inspectedGeneIndices;
    castrate(genome, startGeneIndex, inspectedGeneIndices);
    adaptNodeAttributesForPreview(genome, detailSimulation);
    resetNames(genome);
    if (!detailSimulation) {
        genome._frontAngle = 0;
    }
    if (!genome._genes.empty()) {
        genome._genes.at(startGeneIndex)._numBranches = 1;
    }

    genome._genes.at(startGeneIndex)._separation = true;
    resetUnusedGenes(genome, geneIndices);
}
