#include "GenomeDescriptionEditService.h"

#include <algorithm>
#include <iterator>

#include "GenomeDescriptionInfoService.h"

void GenomeDescriptionEditService::addGene(GenomeDescription& genome, int index, GeneDescription const& newGene)
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

void GenomeDescriptionEditService::removeGene(GenomeDescription& genome, int index)
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

void GenomeDescriptionEditService::swapGenes(GenomeDescription& genome, int index)
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

void GenomeDescriptionEditService::addNode(GeneDescription& gene, int index, NodeDescription const& node)
{
    if (gene._nodes.empty()) {
        gene._nodes.emplace_back(node);
        return;
    }

    gene._nodes.insert(gene._nodes.begin() + index + 1, node);
}

void GenomeDescriptionEditService::removeNode(GeneDescription& gene, int index)
{
    gene._nodes.erase(gene._nodes.begin() + index);
}

void GenomeDescriptionEditService::swapNodes(GeneDescription& gene, int index)
{
    std::swap(gene._nodes.at(index), gene._nodes.at(index + 1));
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
                node._color = 0;
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

void GenomeDescriptionEditService::adaptDescriptionForPreview(GenomeDescription& genome, int startGeneIndex)
{
    castrate(genome, startGeneIndex);
    setNodeAttributesForPreview(genome);
    if (!genome._genes.empty()) {
        genome._genes.at(startGeneIndex)._numBranches = 1;
    }
}
