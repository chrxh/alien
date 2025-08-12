#include "GenomeDescriptionInfoService.h"

#include <algorithm>
#include <iterator>

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

bool GenomeDescriptionInfoService::isConnectedToRoot(GenomeDescription const& genome, int startGeneIndex) const
{
    auto hull = getReferencedGenesInRootGeneHull(genome);
    return hull.contains(startGeneIndex);
}

std::set<int> GenomeDescriptionInfoService::getReferencedGenesInRootGeneHull(GenomeDescription const& genome) const
{
    CHECK(!genome._genes.empty());

    std::set<int> alreadyInspectedGeneIndices = {0};
    std::set<int> toInspectedGeneIndices = alreadyInspectedGeneIndices;
    do {
        std::set<int> newGeneIndices;
        for (auto const& geneIndex : toInspectedGeneIndices) {
            if (geneIndex >= genome._genes.size()) {
                continue;
            }
            auto referenced = getReferences(genome._genes.at(geneIndex));
            newGeneIndices.insert(referenced.begin(), referenced.end());
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

    return alreadyInspectedGeneIndices;
}

auto GenomeDescriptionInfoService::getGeneIndicesForSubGenomes(GenomeDescription const& genome) const -> std::vector<GeneIndicesForSubGenome>
{
    if (genome._genes.empty()) {
        return {};
    }

    std::set<int> nonInspectedGeneIndices;
    for (int i = 0; i < genome._genes.size(); ++i) {
        nonInspectedGeneIndices.insert(i);
    }

    std::vector<GeneIndicesForSubGenome> result;
    while (!nonInspectedGeneIndices.empty()) {
        auto startGeneIndex = *nonInspectedGeneIndices.begin();
        
        auto genesForPart = getGeneIndicesForSubGenomes(genome, startGeneIndex);
        for (auto const& geneIndices : genesForPart) {
            for (auto const& geneIndex : geneIndices) {
                nonInspectedGeneIndices.erase(geneIndex);
            }
        }
        result.insert(result.end(), genesForPart.begin(), genesForPart.end());
    }

    return result;
}

auto GenomeDescriptionInfoService::getGeneIndicesForSubGenomes(GenomeDescription const& genome, int startGeneIndex) const
    -> std::vector<GeneIndicesForSubGenome>
{
    CHECK(!genome._genes.empty());
    CHECK(startGeneIndex >= 0 && startGeneIndex < genome._genes.size());

    std::vector<GeneIndicesForSubGenome> result;

    std::set<int> alreadyInspectedGeneIndices;
    std::set<int> toInspectedGeneIndices = {startGeneIndex};
    do {
        alreadyInspectedGeneIndices.insert(toInspectedGeneIndices.begin(), toInspectedGeneIndices.end());

        std::set<int> newGeneIndices;
        for (auto const& geneIndex : toInspectedGeneIndices) {
            if (geneIndex >= genome._genes.size()) {
                continue;
            }
            ReferencedGenes referenced = getReferencedGenesInNonSeparatingGeneHull(genome, geneIndex);
            if (!referenced.nonSeparatingGeneIndices.empty()) {
                result.emplace_back(referenced.nonSeparatingGeneIndices);
            }
            newGeneIndices.insert(referenced.separatingGeneIndices.begin(), referenced.separatingGeneIndices.end());
        }
        toInspectedGeneIndices.clear();
        std::set_difference(
            newGeneIndices.begin(),
            newGeneIndices.end(),
            alreadyInspectedGeneIndices.begin(),
            alreadyInspectedGeneIndices.end(),
            std::inserter(toInspectedGeneIndices, toInspectedGeneIndices.begin()));

    } while (!toInspectedGeneIndices.empty());

    return result;
}

auto GenomeDescriptionInfoService::getReferencedGenesInNonSeparatingGeneHull(GenomeDescription const& genome, int startGeneIndex) const -> ReferencedGenes
{
    ReferencedGenes result;
    std::set<int> alreadyInspectedGeneIndices;
    std::set<int> toInspectedGeneIndices = {startGeneIndex};
    result.nonSeparatingGeneIndices = {startGeneIndex};
    
    do {
        alreadyInspectedGeneIndices.insert(toInspectedGeneIndices.begin(), toInspectedGeneIndices.end());

        std::set<int> newGeneIndices;
        for (auto const& geneIndex : toInspectedGeneIndices) {
            if (geneIndex >= genome._genes.size()) {
                continue;
            }
            auto referenced = getReferences(genome._genes.at(geneIndex));
            newGeneIndices.insert(referenced.begin(), referenced.end());
        }
        toInspectedGeneIndices.clear();
        std::set_difference(
            newGeneIndices.begin(),
            newGeneIndices.end(),
            alreadyInspectedGeneIndices.begin(),
            alreadyInspectedGeneIndices.end(),
            std::inserter(toInspectedGeneIndices, toInspectedGeneIndices.begin()));
        
        // Separate new genes into separating and non-separating
        std::set<int> newNonSeparatingGenes;
        for (auto const& geneIndex : toInspectedGeneIndices) {
            if (geneIndex < genome._genes.size()) {
                if (genome._genes[geneIndex]._separation) {
                    // This is a separating gene - add to separating list but don't traverse through it
                    result.separatingGeneIndices.push_back(geneIndex);
                } else {
                    // This is a non-separating gene - add to non-separating list and continue traversal
                    result.nonSeparatingGeneIndices.push_back(geneIndex);
                    newNonSeparatingGenes.insert(geneIndex);
                }
            }
        }
        
        // Only add non-separating genes to continue traversal
        toInspectedGeneIndices = newNonSeparatingGenes;
    } while (!toInspectedGeneIndices.empty());

    return result;
}
