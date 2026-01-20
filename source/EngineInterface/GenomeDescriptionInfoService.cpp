#include "GenomeDescriptionInfoService.h"

#include <algorithm>
#include <iterator>

int GenomeDescInfoService::getNumberOfNodes(GenomeDesc const& genome) const
{
    int result = 0;
    for (auto const& gene : genome._genes) {
        result += gene._nodes.size();
    }
    return result;
}

namespace
{
    int countNodes(GenomeDesc const& genome, int geneIndex, std::vector<int>& lastGenes)
    {
        if (std::ranges::find(lastGenes, geneIndex) != lastGenes.end()) {
            return -1;
        }
        if (geneIndex >= genome._genes.size()) {
            return 0;
        }
        lastGenes.emplace_back(geneIndex);

        auto const& gene = genome._genes[geneIndex];
        if (gene._numConcatenations == std::numeric_limits<int>::max()) {
            return -1;
        }
        auto numBranches = gene._separation ? 1 : gene._numBranches;
        auto result = gene._nodes.size() * gene._numConcatenations * numBranches;
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto const& constructor = node._constructor.value();
                if (constructor._geneIndex != 0) {  // First gene is for self-replication and should not be counted
                    auto numNodes = numBranches * countNodes(genome, constructor._geneIndex, lastGenes);
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

int GenomeDescInfoService::getNumberOfResultingCells(GenomeDesc const& genome, int startGeneIndex) const
{
    if (genome._genes.empty()) {
        return 0;
    }
    std::vector<int> lastGenes;
    return countNodes(genome, startGeneIndex, lastGenes);
}

std::vector<int> GenomeDescInfoService::getReferences(GeneDesc const& gene) const
{
    std::vector<int> result;
    for (auto const& node : gene._nodes) {
        if (node._constructor.has_value()) {
            auto const& constructor = node._constructor.value();
            result.emplace_back(constructor._geneIndex);
        }
    }
    return result;
}

std::vector<int> GenomeDescInfoService::getReferencedBy(GenomeDesc const& genome, int geneIndex) const
{
    std::vector<int> result;
    for (int i = 0; i < genome._genes.size(); ++i) {
        auto const& gene = genome._genes[i];
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto const& constructor = node._constructor.value();
                if (constructor._geneIndex == geneIndex) {
                    result.emplace_back(i);
                }
            }
        }
    }
    return result;
}

bool GenomeDescInfoService::isConnectedToRoot(GenomeDesc const& genome, int startGeneIndex) const
{
    auto hull = getReferencedGenesInRootGeneHull(genome);
    return hull.contains(startGeneIndex);
}

std::set<int> GenomeDescInfoService::getReferencedGenesInRootGeneHull(GenomeDesc const& genome) const
{
    if (genome._genes.empty()) {
        return {};
    }

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

auto GenomeDescInfoService::getGeneIndicesForSubGenomes(GenomeDesc const& genome) const -> std::vector<GeneIndicesForSubGenome>
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

        auto genesForPart = getGeneIndicesForSubGenomes(genome, nonInspectedGeneIndices, startGeneIndex);
        for (auto const& geneIndices : genesForPart) {
            for (auto const& geneIndex : geneIndices) {
                nonInspectedGeneIndices.erase(geneIndex);
            }
        }
        result.insert(result.end(), genesForPart.begin(), genesForPart.end());
    }

    return result;
}

auto GenomeDescInfoService::getGeneIndicesForSubGenomes(
    GenomeDesc const& genome,
    std::set<int> const& nonInspectedGeneIndices,
    int startGeneIndex) const -> std::vector<GeneIndicesForSubGenome>
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
            std::vector<int> geneIndices = referenced.nonSeparatingGeneIndices;
            geneIndices.insert(geneIndices.begin(), geneIndex);

            result.emplace_back(geneIndices);

            for (auto const& separatingGeneIndex : referenced.separatingGeneIndices) {
                if (nonInspectedGeneIndices.contains(separatingGeneIndex)) {
                    newGeneIndices.insert(separatingGeneIndex);
                }
            }
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

auto GenomeDescInfoService::getReferencedGenesInNonSeparatingGeneHull(GenomeDesc const& genome, int startGeneIndex) const -> ReferencedGenes
{
    ReferencedGenes result;
    std::set<int> alreadyInspectedGeneIndices;
    std::set<int> toInspectedGeneIndices = {startGeneIndex};

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
