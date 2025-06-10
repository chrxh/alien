#include "GenomeDescriptionEditService.h"

void GenomeDescriptionEditService::addEmptyGene(GenomeDescription_New& genome, int index)
{
    if (genome._genes.empty()) {
        genome._genes.emplace_back(GeneDescription());
        return;
    }

    for (int i = 0; i < genome._genes.size(); ++i) {
        auto& gene = genome._genes[i];
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                if (constructor._constructGeneIndex > index) {
                    ++constructor._constructGeneIndex;
                }
            }
        }
    }

    genome._genes.insert(genome._genes.begin() + index + 1, GeneDescription());
}

void GenomeDescriptionEditService::removeGene(GenomeDescription_New& genome, int index)
{
    for (int i = 0; i < genome._genes.size(); ++i) {
        if (i == index) {
            continue;
        }
        auto& gene = genome._genes[i];
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                if (constructor._constructGeneIndex >= index) {
                    --constructor._constructGeneIndex;
                }
            }
        }
    }
    genome._genes.erase(genome._genes.begin() + index);
}

void GenomeDescriptionEditService::swapGenes(GenomeDescription_New& genome, int index)
{
    std::swap(genome._genes.at(index), genome._genes.at(index + 1));

    for (auto& gene : genome._genes) {
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                if (constructor._constructGeneIndex == index) {
                    constructor._constructGeneIndex = index + 1;
                } else if (constructor._constructGeneIndex == index + 1) {
                    constructor._constructGeneIndex = index;
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
