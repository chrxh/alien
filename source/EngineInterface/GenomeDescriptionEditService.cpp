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
    CollectionDescription&& preview, std::vector<uint64_t> const& creatureIds) const
{
    std::unordered_map<uint64_t, int> creatureIdToIndex;
    for (auto const& [index, creatureId] : creatureIds | boost::adaptors::indexed(0)) {
        creatureIdToIndex.insert_or_assign(creatureId, toInt(index));
    }

    std::vector<CollectionDescription> result(creatureIds.size());
    for (auto& creature : preview._creatures) {
        auto subGenomeIndex = creatureIdToIndex.at(creature._id);
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
                        constructor._geneIndex = genome._genes.size();  // Perform castration
                    } else {
                        castrate(genome, constructor._geneIndex, inspectedGeneIndices);  // Inspect further gene
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

void GenomeDescriptionEditService::adaptDescriptionForPreview(GenomeDescription& genome, int startGeneIndex) const
{
    genome._genes.at(startGeneIndex)._separation = false;

    std::set<int> inspectedGeneIndices;
    castrate(genome, startGeneIndex, inspectedGeneIndices);
    setNodeAttributesForPreview(genome);
    if (!genome._genes.empty()) {
        genome._genes.at(startGeneIndex)._numBranches = 1;
    }
}
