#include "GenomeDescInfoService.h"

#include <algorithm>
#include <iterator>
#include <map>

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
    int countNodes(
        GenomeDesc const& genome,
        int geneIndex,
        bool separation,
        int numBranches,
        int numConcatenations,
        std::vector<int>& lastGenes)
    {
        if (std::ranges::find(lastGenes, geneIndex) != lastGenes.end()) {
            return -1;
        }
        if (geneIndex >= genome._genes.size()) {
            return 0;
        }
        lastGenes.emplace_back(geneIndex);

        auto const& gene = genome._genes[geneIndex];
        if (numConcatenations == std::numeric_limits<int>::max()) {
            return -1;
        }
        auto effectiveNumBranches = separation ? 1 : numBranches;
        auto result = gene._nodes.size() * numConcatenations * effectiveNumBranches;
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto const& constructor = node._constructor.value();
                if (constructor._geneIndex != 0) {  // First gene is for self-replication and should not be counted
                    auto numNodes = effectiveNumBranches
                        * countNodes(
                            genome, constructor._geneIndex, constructor._separation, constructor._numBranches, constructor._numConcatenations, lastGenes);
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

int GenomeDescInfoService::getNumberOfResultingCells(GenomeDesc const& genome, int startGeneIndex, bool separation, int numBranches, int numConcatenations) const
{
    if (genome._genes.empty()) {
        return 0;
    }
    std::vector<int> lastGenes;
    return countNodes(genome, startGeneIndex, separation, numBranches, numConcatenations, lastGenes);
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

auto GenomeDescInfoService::getGeneIndicesForSubGenomes(GenomeDesc const& genome, std::set<int> const& nonInspectedGeneIndices, int startGeneIndex) const
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

        // Collect references along with their constructor separation flag
        std::map<int, bool> newGeneSeparation;  // geneIndex -> isAnyNonSeparating
        for (auto const& geneIndex : toInspectedGeneIndices) {
            if (geneIndex >= genome._genes.size()) {
                continue;
            }
            auto const& gene = genome._genes.at(geneIndex);
            for (auto const& node : gene._nodes) {
                if (node._constructor.has_value()) {
                    auto const& constructor = node._constructor.value();
                    auto targetIndex = constructor._geneIndex;
                    if (alreadyInspectedGeneIndices.contains(targetIndex)) {
                        continue;
                    }
                    auto isNonSeparating = !constructor._separation;
                    auto it = newGeneSeparation.find(targetIndex);
                    if (it == newGeneSeparation.end()) {
                        newGeneSeparation[targetIndex] = isNonSeparating;
                    } else {
                        it->second = it->second || isNonSeparating;  // Treat as non-separating if any reference is non-separating
                    }
                }
            }
        }

        toInspectedGeneIndices.clear();
        std::set<int> newNonSeparatingGenes;
        for (auto const& [geneIndex, isNonSeparating] : newGeneSeparation) {
            if (geneIndex < genome._genes.size()) {
                if (!isNonSeparating) {
                    result.separatingGeneIndices.push_back(geneIndex);
                } else {
                    result.nonSeparatingGeneIndices.push_back(geneIndex);
                    newNonSeparatingGenes.insert(geneIndex);
                }
            }
        }

        toInspectedGeneIndices = newNonSeparatingGenes;
    } while (!toInspectedGeneIndices.empty());

    return result;
}
