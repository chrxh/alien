#include "GenomeDescriptionInfoService.h"

int GenomeDescriptionInfoService::getNumberOfNodes(GenomeDescription const& genome) const
{
    int result = 0;
    for (auto const& gene : genome._genes) {
        result += gene._nodes.size();
    }
    return result;
}

namespace 
{
    int countNodes(GenomeDescription const& genome, int geneIndex, std::vector<int>& lastGenes)
    {
        if (std::ranges::find(lastGenes, geneIndex) != lastGenes.end()) {
            return -1;
        }
        lastGenes.emplace_back(geneIndex);

        auto const& gene = genome._genes[geneIndex];
        if (gene._numConcatenations == std::numeric_limits<int>::max()) {
            return -1;
        }
        auto numBranches = gene._separation ? 1 : gene._numBranches;
        auto result = gene._nodes.size() * gene._numConcatenations * numBranches;
        for (auto const& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto const& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                if (constructor._geneIndex != 0) {  // First gene is for self-replication and should not be counted
                    auto numNodes = countNodes(genome, constructor._geneIndex, lastGenes);
                    if (numNodes == -1) {
                        return -1;  // Cycle detected
                    }
                    result += numNodes;
                }
            }
        }
        lastGenes.pop_back();
        return toInt(result);
    }
}

int GenomeDescriptionInfoService::getNumberOfResultingCells(GenomeDescription const& genome) const
{
    if (genome._genes.empty()) {
        return 0;
    }
    std::vector<int> lastGenes;
    return countNodes(genome, 0, lastGenes);
}

std::vector<int> GenomeDescriptionInfoService::getReferences(GeneDescription const& gene) const
{
    std::vector<int> result;
    for (auto const& node : gene._nodes) {
        if (node.getCellType() == CellTypeGenome_Constructor) {
            auto const& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
            result.emplace_back(constructor._geneIndex);
        }
    }
    return result;
}

std::vector<int> GenomeDescriptionInfoService::getReferencedBy(GenomeDescription const& genome, int geneIndex) const
{
    std::vector<int> result;
    for (int i = 0; i < genome._genes.size(); ++i) {
        auto const& gene = genome._genes[i];
        for (auto const& node : gene._nodes) {
            if (node.getCellType() == CellTypeGenome_Constructor) {
                auto const& constructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                if (constructor._geneIndex == geneIndex) {
                    result.emplace_back(i);
                }
            }
        }
    }
    return result;
}

int GenomeDescriptionInfoService::getRootGene(GenomeDescription const& genome, int geneIndex) const
{
}
