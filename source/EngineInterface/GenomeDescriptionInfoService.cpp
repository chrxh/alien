#include "GenomeDescriptionInfoService.h"

int GenomeDescriptionInfoService::getNumberOfNodes(GenomeDescription_New const& genome) const
{
    int result = 0;
    for (auto const& gene : genome._genes) {
        result += gene._nodes.size();
    }
    return result;
}

namespace 
{
    int countNodes(GenomeDescription_New const& genome, int geneIndex, std::unordered_set<int>& countedGenes)
    {
        if (countedGenes.find(geneIndex) != countedGenes.end()) {
            return -1;
        }
        countedGenes.insert(geneIndex);

        auto const& gene = genome._genes[geneIndex];
        auto result = gene._nodes.size();
        for (auto const& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto const& constructor = std::get<ConstructorGenomeDescription_New>(node._cellTypeData);
                auto numNodes = countNodes(genome, constructor._constructGeneIndex, countedGenes);
                if (numNodes == -1) {
                    return -1;  // Cycle detected
                }
                result += numNodes;
            }
        }
        return toInt(result);
    }
}
int GenomeDescriptionInfoService::getNumberOfResultingCells(GenomeDescription_New const& genome) const
{
    auto result = 0;
    std::unordered_set<int> countedGenes;
    for (int i = 0; i < genome._genes.size(); ++i) {
        auto numNodes = countNodes(genome, i, countedGenes);
        if (numNodes == -1) {
            return -1; // Cycle detected
        }
        result += numNodes;
    }
    return result;
}
