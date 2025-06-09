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
