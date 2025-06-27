#include "CreatureDescriptionEditService.h"

void CreatureDescriptionEditService::addGene(CreatureDescription& creature, int index, GeneDescription const& newGene)
{
    if (creature._genes.empty()) {
        creature._genes.emplace_back(newGene);
        return;
    }

    for (int i = 0; i < creature._genes.size(); ++i) {
        auto& gene = creature._genes[i];
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                if (constructor._geneIndex > index) {
                    ++constructor._geneIndex;
                }
            }
        }
    }

    creature._genes.insert(creature._genes.begin() + index + 1, newGene);
}

void CreatureDescriptionEditService::removeGene(CreatureDescription& creature, int index)
{
    for (int i = 0; i < creature._genes.size(); ++i) {
        if (i == index) {
            continue;
        }
        auto& gene = creature._genes[i];
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                if (constructor._geneIndex >= index) {
                    --constructor._geneIndex;
                }
            }
        }
    }
    creature._genes.erase(creature._genes.begin() + index);
}

void CreatureDescriptionEditService::swapGenes(CreatureDescription& creature, int index)
{
    std::swap(creature._genes.at(index), creature._genes.at(index + 1));

    for (auto& gene : creature._genes) {
        for (auto& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                if (constructor._geneIndex == index) {
                    constructor._geneIndex = index + 1;
                } else if (constructor._geneIndex == index + 1) {
                    constructor._geneIndex = index;
                }
            }
        }
    }
}

void CreatureDescriptionEditService::addEmptyNode(GeneDescription& gene, int index)
{
    if (gene._nodes.empty()) {
        gene._nodes.emplace_back(NodeDescription());
        return;
    }

    gene._nodes.insert(gene._nodes.begin() + index + 1, NodeDescription());
}

void CreatureDescriptionEditService::removeNode(GeneDescription& gene, int index)
{
    gene._nodes.erase(gene._nodes.begin() + index);
}

void CreatureDescriptionEditService::swapNodes(GeneDescription& gene, int index)
{
    std::swap(gene._nodes.at(index), gene._nodes.at(index + 1));
}
