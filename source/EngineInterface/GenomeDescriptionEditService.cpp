#include "GenomeDescriptionEditService.h"

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

void GenomeDescriptionEditService::addEmptyNode(GeneDescription& gene, int index)
{
    if (gene._nodes.empty()) {
        gene._nodes.emplace_back(NodeDescription());
        return;
    }

    gene._nodes.insert(gene._nodes.begin() + index + 1, NodeDescription());
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
    void castrateIntern(GenomeDescription& genome, int geneIndex, std::set<int>& inspectedGeneIndices)
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
                    auto const& gene = genome._genes.at(constructor._geneIndex);
                    if (inspectedGeneIndices.contains(constructor._geneIndex) && gene._separation) {
                        constructor._geneIndex = genome._genes.size();  // Perform castration
                    } else {
                        castrateIntern(genome, constructor._geneIndex, inspectedGeneIndices);   // Inspect further gene
                    }
                }
            }
        }
    }
}

void GenomeDescriptionEditService::castrate(GenomeDescription& genome)
{
    std::set<int> inspectedGeneIndices;
    castrateIntern(genome, 0, inspectedGeneIndices);
}
