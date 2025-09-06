#include "GenomeDescriptionEditService.h"

#include <algorithm>
#include <iterator>

#include <boost/range/adaptors.hpp>

#include "DescriptionEditService.h"
#include "GenomeDescriptionInfoService.h"

namespace 
{
    auto constexpr PreviewColor = 0;
}

void GenomeDescriptionEditService::addGene(GenomeDescription& genome, int index, GeneDescription const& newGene) const
{
    if (genome._genes.empty()) {
        genome._genes.emplace_back(newGene);
        return;
    }

    for (int i = 0; i < genome._genes.size(); ++i) {
        auto& gene = genome._genes[i];
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                if (constructor._geneIndex > index) {
                    ++constructor._geneIndex;
                }
            }
        }
    }

    genome._genes.insert(genome._genes.begin() + index + 1, newGene);
}

void GenomeDescriptionEditService::removeGene(GenomeDescription& genome, int index) const
{
    for (int i = 0; i < genome._genes.size(); ++i) {
        if (i == index) {
            continue;
        }
        auto& gene = genome._genes[i];
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                if (constructor._geneIndex >= index) {
                    --constructor._geneIndex;
                }
            }
        }
    }
    genome._genes.erase(genome._genes.begin() + index);
}

void GenomeDescriptionEditService::swapGenes(GenomeDescription& genome, int index) const
{
    std::swap(genome._genes.at(index), genome._genes.at(index + 1));

    for (auto& gene : genome._genes) {
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                if (constructor._geneIndex == index) {
                    constructor._geneIndex = index + 1;
                } else if (constructor._geneIndex == index + 1) {
                    constructor._geneIndex = index;
                }
            }
        }
    }
}

void GenomeDescriptionEditService::addNode(GeneDescription& gene, int index, NodeDescription const& node) const
{
    if (gene._nodes.empty()) {
        gene._nodes.emplace_back(node);
        return;
    }

    gene._nodes.insert(gene._nodes.begin() + index + 1, node);
}

void GenomeDescriptionEditService::removeNode(GeneDescription& gene, int index) const
{
    gene._nodes.erase(gene._nodes.begin() + index);
}

void GenomeDescriptionEditService::swapNodes(GeneDescription& gene, int index) const
{
    std::swap(gene._nodes.at(index), gene._nodes.at(index + 1));
}

std::vector<GenomeDescriptionWithStartGeneIndex> GenomeDescriptionEditService::createSubGenomesForPreview(
    GenomeDescription const& genome,
    std::vector<GeneIndicesForSubGenome> const& geneIndicesForSubGenomes) const
{
    std::vector<GenomeDescriptionWithStartGeneIndex> result;
    for (auto const& geneIndicesForSubGenome : geneIndicesForSubGenomes) {
        std::set creatureGeneSet(geneIndicesForSubGenome.begin(), geneIndicesForSubGenome.end());
        auto clone = genome;
        adaptDescriptionForPreview(clone, geneIndicesForSubGenome.front());
        for (int i = 0, size = clone._genes.size(); i < size; ++i) {
            if (!creatureGeneSet.contains(i)) {
                clone._genes[i] = GeneDescription();
            }
        }
        result.emplace_back(clone, geneIndicesForSubGenome.front());
    }
    return result;
}

auto GenomeDescriptionEditService::createSeedCollectionForPreview(
    std::vector<GenomeDescriptionWithStartGeneIndex> const& subGenomes,
    std::unordered_map<GenomeDescriptionWithStartGeneIndex, CollectionDescription> const& cache) const -> SeedCollectionResult
{
    auto const& editService = DescriptionEditService::get();

    RealVector2D currentPos{toFloat(PREVIEW_HEIGHT) / 2, toFloat(PREVIEW_HEIGHT) / 2};

    SeedCollectionResult result;
    for (auto const& subGenome : subGenomes) {
        auto findResult = cache.find(subGenome);
        if (findResult != cache.end()) {
            auto cachedPhenotype = findResult->second;
            editService.setCenter(cachedPhenotype, currentPos);

            CHECK(cachedPhenotype._creatures.size() <= 2);
            auto seedFirst = false;
            if (cachedPhenotype._creatures.front()._generation == 0) {
                seedFirst = true;  // first Creature is seed
            }
            result.data.add(std::move(cachedPhenotype));

            auto index =
                seedFirst ? result.data._creatures.size() - cachedPhenotype._creatures.size() : result.data._creatures.size() - cachedPhenotype._creatures.size() + 1;
            result.seedCreatureIds.emplace_back(result.data._creatures.at(index)._id);
        } else {
            auto seed = createSeedForPreview(subGenome, currentPos);
            result.data.add(std::move(seed));

            result.seedCreatureIds.emplace_back(result.data._creatures.back()._id);
        }
        currentPos.x += toFloat(PREVIEW_HEIGHT) / 2;  // Adjust position for the next sub-genome
    }
    return result;
}

std::vector<CollectionDescription> GenomeDescriptionEditService::extractPhenotypesFromPreview(
    CollectionDescription&& preview, std::vector<uint64_t> const& seedCreatureIds) const
{
    std::unordered_map<uint64_t, int> creatureIdToIndex;
    for (auto const& [index, creatureId] : seedCreatureIds | boost::adaptors::indexed(0)) {
        creatureIdToIndex.insert_or_assign(creatureId, toInt(index));
    }

    std::vector<CollectionDescription> result(seedCreatureIds.size());
    for (auto& creature : preview._creatures) {
        if (creature._generation == 0) {
            auto subGenomeIndex = creatureIdToIndex.at(creature._id);
            result.at(subGenomeIndex)._creatures.emplace_back(std::move(creature));
        } else {
            CHECK(creature._generation == 1);
            auto subGenomeIndex = creatureIdToIndex.at(creature._ancestorId.value());
            result.at(subGenomeIndex)._creatures.emplace_back(std::move(creature));
        }
    }
    return result;
}

void GenomeDescriptionEditService::removeSeedFromPhenotype(CollectionDescription& phenotype) const
{
    std::set<uint64_t> seedCellIds;
    for (auto const& creature : phenotype._creatures) {
        if (creature._generation == 0) {
            for (auto const& cell : creature._cells) {
                seedCellIds.insert(cell._id);
            }
        }
    }
    DescriptionEditService::get().removeCellIf(phenotype, [&seedCellIds](auto const& cell) { return seedCellIds.contains(cell._id); });
}

namespace
{
    void castrate(GenomeDescription& genome, int geneIndex, std::set<int>& inspectedGeneIndices)
    {
        if (geneIndex >= genome._genes.size() || inspectedGeneIndices.contains(geneIndex)) {
            return;
        }
        inspectedGeneIndices.insert(geneIndex);
        auto& gene = genome._genes.at(geneIndex);
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
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

    void setNodeAttributesForPreview(GenomeDescription& genome)
    {
        for (auto& gene : genome._genes) {
            for (auto& node : gene._nodes) {
                node._color = PreviewColor;
                node._neuralNetwork = NeuralNetworkGenomeDescription();
                node._signalRestriction = SignalRestrictionGenomeDescription();
                if (node.getCellType() != CellTypeGenome_Constructor) {
                    node._cellTypeData = BaseGenomeDescription();
                } else {
                    auto& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                    constructor._autoTriggerInterval = 75;
                    constructor._constructionActivationTime = 200;
                }
            }
        }
    }
}

CollectionDescription GenomeDescriptionEditService::createSeedForPreview(GenomeDescriptionWithStartGeneIndex const& subGenome, RealVector2D const& pos) const
{
    return CollectionDescription().creatures({
        CreatureDescription()
            .genome(subGenome.genome)
            .cells({
                CellDescription().color(PreviewColor).stiffness(1.0f).cellTypeData(ConstructorDescription().geneIndex(subGenome.startIndex)).pos(pos),
            }),
    });
}

void GenomeDescriptionEditService::adaptDescriptionForPreview(GenomeDescription& genome, int startGeneIndex) const
{
    std::set<int> inspectedGeneIndices;
    castrate(genome, startGeneIndex, inspectedGeneIndices);
    setNodeAttributesForPreview(genome);
    genome._frontAngle = 0;
    if (!genome._genes.empty()) {
        genome._genes.at(startGeneIndex)._numBranches = 1;
    }

    genome._genes.at(startGeneIndex)._separation = true;
}
