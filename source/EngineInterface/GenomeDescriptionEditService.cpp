#include "GenomeDescriptionEditService.h"

#include <algorithm>
#include <iterator>

#include <boost/range/adaptors.hpp>

#include "DescriptionEditService.h"
#include "GenomeDescriptionInfoService.h"

namespace 
{
    auto constexpr PreviewColor = 0;
    auto constexpr SeedColor = 1;
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
        for (int i = 0, size = clone._genes.size(); i < size; ++i) {
            if (!creatureGeneSet.contains(i)) {
                clone._genes[i] = GeneDescription();
            }
        }
        adaptDescriptionForPreview(clone, geneIndicesForSubGenome.front());
        result.emplace_back(clone, geneIndicesForSubGenome.front());
    }
    return result;
}

CollectionDescription GenomeDescriptionEditService::createSeedForPreview(
    GenomeDescriptionWithStartGeneIndex const& genomeWithStartIndex,
    RealVector2D const& pos) const
{
    return CollectionDescription().creatures({
        CreatureDescription()
            .genome(genomeWithStartIndex.genome)
            .cells({
                CellDescription().color(SeedColor).stiffness(1.0f).cellTypeData(ConstructorDescription().geneIndex(genomeWithStartIndex.startIndex)).pos(pos),
            }),
    });
}

std::vector<CollectionDescription> GenomeDescriptionEditService::extractPhenotypesFromPreview(
    CollectionDescription&& preview,
    std::vector<GenomeDescriptionWithStartGeneIndex> const& subGenomes) const
{
    // Calc startGeneIndex to subGenomeIndex map
    std::unordered_map<int, int> startGeneIndexToSubGenomeIndex;
    for (auto const& [index, subGenome] : subGenomes | boost::adaptors::indexed(0)) {
        startGeneIndexToSubGenomeIndex.insert_or_assign(subGenome.startIndex, toInt(index));
    }

    // Calc creature id to each sub-genome
    std::unordered_map<uint64_t, int> creatureIdToSubGenomeIndex;
    for (auto const& creature : preview._creatures) {
        for (auto const& cell : creature._cells) {
            if (cell._color != PreviewColor) {
                continue;
            }
            auto findResult = startGeneIndexToSubGenomeIndex.find(cell._geneIndex);
            if (findResult != startGeneIndexToSubGenomeIndex.end()) {
                auto subGenomeIndex = findResult->second;
                creatureIdToSubGenomeIndex.insert_or_assign(creature._id, subGenomeIndex);
            }
        }
    }

    // Move constructed creatures to result vector
    std::vector<CollectionDescription> result(subGenomes.size());
    for (auto& creature : preview._creatures) {
        auto subGenomeIndex = creatureIdToSubGenomeIndex.at(creature._id);
        result.at(subGenomeIndex)._creatures.emplace_back(std::move(creature));
    }

    return result;
}

void GenomeDescriptionEditService::removeSeedFromPhenotype(CollectionDescription& phenotype) const
{
    DescriptionEditService::get().removeCellIf(phenotype, [](auto const& cell) { return cell._color == SeedColor; });
}

namespace
{
    void castrate(GenomeDescription& genome, int startGeneIndex)
    {
        std::set<int> alreadyInspectedGeneIndices = {startGeneIndex};
        std::set<int> toInspectedGeneIndices = alreadyInspectedGeneIndices;
        do {
            std::set<int> newGeneIndices;
            for (auto const& geneIndex : toInspectedGeneIndices) {
                if (geneIndex >= genome._genes.size()) {
                    continue;
                }
                std::vector<int> referencedGeneIndices;
                auto& gene = genome._genes.at(geneIndex);
                for (auto& node : gene._nodes) {
                    if (node.getCellType() == CellTypeGenome_Constructor) {
                        auto& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                        referencedGeneIndices.emplace_back(constructor._geneIndex);
                        if (alreadyInspectedGeneIndices.contains(constructor._geneIndex)) {
                            constructor._geneIndex = genome._genes.size();  // Perform castration
                        }
                    }
                }

                newGeneIndices.insert(referencedGeneIndices.begin(), referencedGeneIndices.end());
            }
            toInspectedGeneIndices.clear();
            std::set_difference(
                newGeneIndices.begin(),
                newGeneIndices.end(),
                alreadyInspectedGeneIndices.begin(),
                alreadyInspectedGeneIndices.end(),
                std::inserter(toInspectedGeneIndices, toInspectedGeneIndices.begin()));
            alreadyInspectedGeneIndices.insert(newGeneIndices.begin(), newGeneIndices.end());
        } while (!toInspectedGeneIndices.empty());
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
                    constructor._constructionActivationTime = 100;
                }
            }
        }
    }
}

void GenomeDescriptionEditService::adaptDescriptionForPreview(GenomeDescription& genome, int startGeneIndex) const
{
    genome._genes.at(startGeneIndex)._separation = false;
    castrate(genome, startGeneIndex);
    setNodeAttributesForPreview(genome);
    if (!genome._genes.empty()) {
        genome._genes.at(startGeneIndex)._numBranches = 1;
    }
}
