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
        int64_t result = gene._nodes.size();
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value()) {
                auto const& constructor = node._constructor.value();
                if (constructor._geneIndex != 0) {  // First gene is for self-replication and should not be counted
                    auto numNodes = countNodes(genome, constructor._geneIndex, lastGenes);
                    if (numNodes == -1) {
                        return -1;  // Cycle detected
                    }
                    if (constructor._numConcatenations == std::numeric_limits<int>::max() && numNodes > 0) {
                        return -1;
                    }
                    auto effectiveNumBranches = constructor._separation ? 1 : constructor._numBranches;
                    result += effectiveNumBranches * constructor._numConcatenations * numNodes;
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

        ReferencedGenes referenced = getReferencedGenesInNonSeparatingGeneHull(genome, startGeneIndex);
        GeneIndicesForSubGenome geneIndices = referenced.nonSeparatingGeneIndices;
        geneIndices.insert(geneIndices.begin(), startGeneIndex);

        for (auto const& geneIndex : geneIndices) {
            nonInspectedGeneIndices.erase(geneIndex);
        }
        result.emplace_back(geneIndices);
    }

    return dropContainedSubGenomes(genome, result);
}

auto GenomeDescInfoService::dropContainedSubGenomes(GenomeDesc const& genome, std::vector<GeneIndicesForSubGenome> const& subGenomes) const
    -> std::vector<GeneIndicesForSubGenome>
{
    // Sub-genomes that are kept even when fully contained in another: the root sub-genome and any sub-genome whose
    // start gene is separatingly referenced by some node (such a reference starts a new creature).
    std::set<int> separatinglyReferencedGenes;
    for (auto const& gene : genome._genes) {
        for (auto const& node : gene._nodes) {
            if (node._constructor.has_value() && node._constructor->_separation) {
                separatinglyReferencedGenes.insert(node._constructor->_geneIndex);
            }
        }
    }

    std::vector<std::set<int>> geneSets;
    for (auto const& geneIndices : subGenomes) {
        geneSets.emplace_back(geneIndices.begin(), geneIndices.end());
    }

    std::vector<GeneIndicesForSubGenome> result;
    for (int i = 0, size = toInt(subGenomes.size()); i < size; ++i) {
        auto startGeneIndex = subGenomes[i].front();
        bool keepAlways = startGeneIndex == 0 || separatinglyReferencedGenes.contains(startGeneIndex);

        bool containedInOther = false;
        if (!keepAlways) {
            for (int j = 0; j < size; ++j) {
                if (i != j && geneSets[i].size() < geneSets[j].size()
                    && std::includes(geneSets[j].begin(), geneSets[j].end(), geneSets[i].begin(), geneSets[i].end())) {
                    containedInOther = true;
                    break;
                }
            }
        }
        if (!containedInOther) {
            result.emplace_back(subGenomes[i]);
        }
    }

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
