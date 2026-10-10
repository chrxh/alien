#include "CpuGeneGraphProcessor.h"

#include <algorithm>
#include <ranges>
#include <set>
#include <numeric>
#include <span>

#include <boost/range/adaptor/indexed.hpp>

#include <Base/Interface/Definitions.h>

#include "GenomeDescAccessService.h"
#include "ShapeGenerator.h"

namespace
{
    struct SearchStackEntry
    {
        int geneIndex = 0;
        int nodeIndex = 0;
    };

    bool hasStructuralNodeMutations(MutationRatesDesc const& rates)
    {
        return rates._extendGeneMutation._geneProbability > 0 || rates._addNodeMutation._nodeProbability > 0 || rates._trimGeneMutation._geneProbability > 0
            || rates._deleteNodeMutation._nodeProbability > 0 || rates._copyNodeSectionMutation._geneProbability > 0
            || rates._moveNodeSectionMutation._geneProbability > 0;
    }
}

CpuGeneGraphProcessor::CpuGeneGraphProcessor(GenomeDesc const& genome)
    : _genome(genome)
    , _correctsVoidBoundaryNodes(hasStructuralNodeMutations(genome._mutationRates))
{
    for (auto const& [geneIndex, gene] : genome._genes | boost::adaptors::indexed(0)) {
        GeneState geneState{.geneIndex = toInt(geneIndex), .shape = gene._shape, .homogeneousCellType = gene._homogeneousCellType};
        for (auto const& node : gene._nodes) {
            NodeState nodeState{.isVoid = node.getCellType() == CellType_Void};
            if (node._constructor.has_value() && isValidGeneIndex(node._constructor->_geneIndex)) {
                nodeState.constructorGeneIndex = node._constructor->_geneIndex;
                nodeState.separation = node._constructor->_separation;
            }
            if (auto injector = std::get_if<InjectorGenomeDesc>(&node._cellType)) {
                nodeState.injectorGeneIndex = injector->_geneIndex;
            }
            geneState.nodes.emplace_back(nodeState);
        }
        _genes.emplace_back(geneState);
    }
}

std::vector<GenomeIssue> CpuGeneGraphProcessor::process()
{
    if (_genes.empty()) {
        return {};
    }
    if (_correctsVoidBoundaryNodes) {
        correctVoidBoundaryNodes();
    }
    voidNodesCutOffFromLastNode();
    removeCyclesAvoidingRootGene();
    removeGenesUnreachableFromRoot();
    limitGenesWithSeparation();
    return _issues;
}

std::vector<int> CpuGeneGraphProcessor::calcNodesCutOffFromLastNode(GeneState const& gene)
{
    auto numNodes = toInt(gene.nodes.size());
    if (gene.homogeneousCellType || numNodes < 2 || std::ranges::none_of(gene.nodes, &NodeState::isVoid)) {
        return {};
    }

    std::vector<int> components(numNodes);
    std::iota(components.begin(), components.end(), 0);
    auto findComponent = [&](int nodeIndex) {
        while (components.at(nodeIndex) != nodeIndex) {
            components.at(nodeIndex) = components.at(components.at(nodeIndex));
            nodeIndex = components.at(nodeIndex);
        }
        return nodeIndex;
    };
    auto connect = [&](int nodeIndex1, int nodeIndex2) {
        auto component1 = findComponent(nodeIndex1);
        auto component2 = findComponent(nodeIndex2);
        if (component1 != component2) {
            components.at(component2) = component1;
        }
    };
    auto isVoid = [&](int nodeIndex) { return gene.nodes.at(nodeIndex).isVoid; };

    ShapeGenerator shapeGenerator;
    for (auto nodeIndex : std::views::iota(0, numNodes)) {
        auto shapeResult = shapeGenerator.generateNextConstructionData(gene.shape);
        if (isVoid(nodeIndex)) {
            continue;
        }
        if (nodeIndex > 0 && !isVoid(nodeIndex - 1)) {
            connect(nodeIndex, nodeIndex - 1);
        }
        for (auto otherNodeIndex : std::span(shapeResult.requiredNodeId, shapeResult.numAdditionalConnections)) {
            if (otherNodeIndex >= 0 && otherNodeIndex < nodeIndex && !isVoid(otherNodeIndex)) {
                connect(nodeIndex, otherNodeIndex);
            }
        }
    }

    std::vector<int> result;
    auto lastComponent = findComponent(numNodes - 1);
    for (auto nodeIndex : std::views::iota(0, numNodes)) {
        if (findComponent(nodeIndex) != lastComponent) {
            result.emplace_back(nodeIndex);
        }
    }
    return result;
}

void CpuGeneGraphProcessor::correctVoidBoundaryNodes()
{
    for (auto& gene : _genes) {
        if (gene.nodes.empty()) {
            continue;
        }
        for (auto nodeIndex : std::set{0, toInt(gene.nodes.size()) - 1}) {
            auto& node = gene.nodes.at(nodeIndex);
            if (!node.isVoid) {
                continue;
            }
            node.isVoid = false;

            auto isExpressedAsFirstNode = gene.homogeneousCellType && nodeIndex > 0;
            if (!isExpressedAsFirstNode) {
                _issues.emplace_back(GenomeIssue{.type = GenomeIssueType::VoidBoundaryNode, .geneIndex = gene.geneIndex, .nodeIndex = nodeIndex});
            }
        }
    }
}

void CpuGeneGraphProcessor::voidNodesCutOffFromLastNode()
{
    std::map<int, std::vector<int>> cutOffNodeIndicesByGeneIndex;
    for (auto const& gene : _genes) {
        cutOffNodeIndicesByGeneIndex.emplace(gene.geneIndex, calcNodesCutOffFromLastNode(gene));
    }
    auto isCutOff = [&](int geneIndex, int nodeIndex) {
        auto const& cutOffNodeIndices = cutOffNodeIndicesByGeneIndex.at(geneIndex);
        return std::ranges::find(cutOffNodeIndices, nodeIndex) != cutOffNodeIndices.end();
    };
    std::erase_if(_issues, [&](GenomeIssue const& issue) {
        return issue.type == GenomeIssueType::VoidBoundaryNode && isCutOff(issue.geneIndex, issue.nodeIndex.value());
    });

    auto losesFirstNode = [](std::vector<int> const& cutOffNodeIndices) { return !cutOffNodeIndices.empty() && cutOffNodeIndices.front() == 0; };
    auto keepsFirstGene = std::ranges::all_of(cutOffNodeIndicesByGeneIndex | std::views::values, losesFirstNode);

    std::map<int, int> removalIssueIndexByGeneIndex;
    for (auto& gene : _genes) {
        auto const& cutOffNodeIndices = cutOffNodeIndicesByGeneIndex.at(gene.geneIndex);
        for (auto nodeIndex : cutOffNodeIndices) {
            gene.nodes.at(nodeIndex) = NodeState{.isVoid = true};
        }

        auto removesGene = losesFirstNode(cutOffNodeIndices) && !(keepsFirstGene && gene.geneIndex == _genes.front().geneIndex);
        auto changesAnyNode = std::ranges::any_of(cutOffNodeIndices, [&](int nodeIndex) { return isChangedByVoiding(gene.geneIndex, nodeIndex); });
        if (!removesGene && !changesAnyNode) {
            continue;
        }

        GenomeIssue issue{
            .type = GenomeIssueType::NodesCutOffFromLastNode, .geneIndex = gene.geneIndex, .cutOffNodeIndices = cutOffNodeIndices, .removesGene = removesGene};
        if (!_correctsVoidBoundaryNodes) {
            auto const& nodes = _genome._genes.at(gene.geneIndex)._nodes;
            if (nodes.front().getCellType() == CellType_Void) {
                issue.nodeIndex = 0;
            } else if (nodes.back().getCellType() == CellType_Void) {
                issue.nodeIndex = toInt(nodes.size()) - 1;
            }
        }
        if (removesGene) {
            removalIssueIndexByGeneIndex.emplace(gene.geneIndex, toInt(_issues.size()));
        }
        _issues.emplace_back(issue);
    }
    removeGenes(removalIssueIndexByGeneIndex);
}

void CpuGeneGraphProcessor::removeCyclesAvoidingRootGene()
{
    enum class SearchState
    {
        NotVisited,
        OnStack,
        Finished
    };
    auto positions = getGenePositions();
    std::vector<SearchState> states(_genome._genes.size(), SearchState::NotVisited);
    states.at(_genes.front().geneIndex) = SearchState::Finished;

    for (auto const& startGene : _genes | std::views::drop(1)) {
        if (states.at(startGene.geneIndex) != SearchState::NotVisited) {
            continue;
        }
        states.at(startGene.geneIndex) = SearchState::OnStack;
        std::vector<SearchStackEntry> stack = {{.geneIndex = startGene.geneIndex}};

        while (!stack.empty()) {
            auto [geneIndex, nodeIndex] = stack.back();
            auto& gene = _genes.at(positions.at(geneIndex));
            if (nodeIndex >= toInt(gene.nodes.size())) {
                states.at(geneIndex) = SearchState::Finished;
                stack.pop_back();
                continue;
            }
            ++stack.back().nodeIndex;

            auto& node = gene.nodes.at(nodeIndex);
            if (!node.constructorGeneIndex.has_value()) {
                continue;
            }
            auto targetGeneIndex = node.constructorGeneIndex.value();
            if (states.at(targetGeneIndex) == SearchState::OnStack) {
                GenomeIssue issue{.type = GenomeIssueType::CycleAvoidingRootGene, .geneIndex = geneIndex, .nodeIndex = nodeIndex};
                auto cycleStart = std::ranges::find(stack, targetGeneIndex, &SearchStackEntry::geneIndex);
                for (auto const& entry : std::ranges::subrange(cycleStart, stack.end())) {
                    issue.cycleGeneIndices.emplace_back(entry.geneIndex);
                }
                issue.cycleGeneIndices.emplace_back(targetGeneIndex);
                _issues.emplace_back(issue);
                node.constructorGeneIndex.reset();
            } else if (states.at(targetGeneIndex) == SearchState::NotVisited) {
                states.at(targetGeneIndex) = SearchState::OnStack;
                stack.emplace_back(SearchStackEntry{.geneIndex = targetGeneIndex});
            }
        }
    }
}

void CpuGeneGraphProcessor::removeGenesUnreachableFromRoot()
{
    auto positions = getGenePositions();
    std::set<int> reachedGeneIndices = {_genes.front().geneIndex};
    std::vector<int> genesToScan = {_genes.front().geneIndex};
    while (!genesToScan.empty()) {
        auto geneIndex = genesToScan.back();
        genesToScan.pop_back();
        for (auto const& node : _genes.at(positions.at(geneIndex)).nodes) {
            if (node.constructorGeneIndex.has_value() && reachedGeneIndices.insert(node.constructorGeneIndex.value()).second) {
                genesToScan.emplace_back(node.constructorGeneIndex.value());
            }
        }
    }

    std::map<int, int> removalIssueIndexByGeneIndex;
    for (auto const& gene : _genes) {
        if (reachedGeneIndices.contains(gene.geneIndex)) {
            continue;
        }
        auto issue = GenomeIssue{
            .type = GenomeIssueType::GeneUnreachableFromRoot,
            .geneIndex = gene.geneIndex,
            .removesGene = true,
            .causeIssueIndex = findCauseOfUnreachability(gene.geneIndex)};
        removalIssueIndexByGeneIndex.emplace(gene.geneIndex, toInt(_issues.size()));
        _issues.emplace_back(issue);
    }
    removeGenes(removalIssueIndexByGeneIndex);
}

void CpuGeneGraphProcessor::limitGenesWithSeparation()
{
    auto positions = getGenePositions();
    std::vector<int> genesWithSeparation;
    std::set<int> visitedGeneIndices = {_genes.front().geneIndex};
    std::vector<SearchStackEntry> stack = {{.geneIndex = _genes.front().geneIndex}};

    while (!stack.empty()) {
        auto [geneIndex, nodeIndex] = stack.back();
        auto& gene = _genes.at(positions.at(geneIndex));
        if (nodeIndex >= toInt(gene.nodes.size())) {
            stack.pop_back();
            continue;
        }
        ++stack.back().nodeIndex;

        auto& node = gene.nodes.at(nodeIndex);
        if (!node.constructorGeneIndex.has_value()) {
            continue;
        }
        auto targetGeneIndex = node.constructorGeneIndex.value();
        if (node.separation && std::ranges::find(genesWithSeparation, targetGeneIndex) == genesWithSeparation.end()) {
            if (toInt(genesWithSeparation.size()) < Const::MaxGenesWithSeparation) {
                genesWithSeparation.emplace_back(targetGeneIndex);
            } else {
                _issues.emplace_back(GenomeIssue{
                    .type = GenomeIssueType::TooManyGenesWithSeparation,
                    .geneIndex = geneIndex,
                    .nodeIndex = nodeIndex,
                    .geneIndicesKeepingSeparation = genesWithSeparation});
                node.separation = false;
            }
        }
        if (visitedGeneIndices.insert(targetGeneIndex).second) {
            stack.emplace_back(SearchStackEntry{.geneIndex = targetGeneIndex});
        }
    }
}

void CpuGeneGraphProcessor::removeGenes(std::map<int, int> const& removalIssueIndexByGeneIndex)
{
    if (removalIssueIndexByGeneIndex.empty()) {
        return;
    }
    std::erase_if(_genes, [&](GeneState const& gene) { return removalIssueIndexByGeneIndex.contains(gene.geneIndex); });

    for (auto& gene : _genes) {
        for (auto const& [nodeIndex, node] : gene.nodes | boost::adaptors::indexed(0)) {
            if (node.constructorGeneIndex.has_value() && removalIssueIndexByGeneIndex.contains(node.constructorGeneIndex.value())) {
                _issues.emplace_back(GenomeIssue{
                    .type = GenomeIssueType::ConstructsRemovedGene,
                    .geneIndex = gene.geneIndex,
                    .nodeIndex = toInt(nodeIndex),
                    .referencedGeneIndex = node.constructorGeneIndex.value(),
                    .causeIssueIndex = removalIssueIndexByGeneIndex.at(node.constructorGeneIndex.value())});
                node.constructorGeneIndex.reset();
            }
            if (node.injectorGeneIndex.has_value() && removalIssueIndexByGeneIndex.contains(node.injectorGeneIndex.value())) {
                _issues.emplace_back(GenomeIssue{
                    .type = GenomeIssueType::InjectsRemovedGene,
                    .geneIndex = gene.geneIndex,
                    .nodeIndex = toInt(nodeIndex),
                    .referencedGeneIndex = node.injectorGeneIndex.value(),
                    .causeIssueIndex = removalIssueIndexByGeneIndex.at(node.injectorGeneIndex.value())});
                node.injectorGeneIndex = _genes.front().geneIndex;
            }
        }
    }
}

std::optional<int> CpuGeneGraphProcessor::findCauseOfUnreachability(int geneIndex) const
{
    auto const& genomeAccessService = GenomeDescAccessService::get();
    if (!genomeAccessService.getReachableGenes(_genome, 0).contains(geneIndex)) {
        return std::nullopt;
    }
    for (auto const& [issueIndex, issue] : _issues | boost::adaptors::indexed(0)) {
        for (auto disconnectedGeneIndex : getDisconnectedGeneIndices(issue)) {
            if (genomeAccessService.getReachableGenes(_genome, disconnectedGeneIndex).contains(geneIndex)) {
                return toInt(issueIndex);
            }
        }
    }
    return std::nullopt;
}

std::vector<int> CpuGeneGraphProcessor::getDisconnectedGeneIndices(GenomeIssue const& issue) const
{
    auto const& nodes = _genome._genes.at(issue.geneIndex)._nodes;
    std::vector<int> nodeIndices;
    if (issue.removesGene) {
        nodeIndices.resize(nodes.size());
        std::iota(nodeIndices.begin(), nodeIndices.end(), 0);
    } else if (issue.type == GenomeIssueType::NodesCutOffFromLastNode) {
        nodeIndices = issue.cutOffNodeIndices;
    } else if (issue.type == GenomeIssueType::CycleAvoidingRootGene) {
        nodeIndices = {issue.nodeIndex.value()};
    }

    std::vector<int> result;
    for (auto nodeIndex : nodeIndices) {
        auto const& constructor = nodes.at(nodeIndex)._constructor;
        if (constructor.has_value() && isValidGeneIndex(constructor->_geneIndex)) {
            result.emplace_back(constructor->_geneIndex);
        }
    }
    return result;
}

bool CpuGeneGraphProcessor::isChangedByVoiding(int geneIndex, int nodeIndex) const
{
    auto const& node = _genome._genes.at(geneIndex)._nodes.at(nodeIndex);
    return node.getCellType() != CellType_Void || node._constructor.has_value();
}

std::vector<int> CpuGeneGraphProcessor::getGenePositions() const
{
    std::vector<int> result(_genome._genes.size(), -1);
    for (auto const& [position, gene] : _genes | boost::adaptors::indexed(0)) {
        result.at(gene.geneIndex) = toInt(position);
    }
    return result;
}

bool CpuGeneGraphProcessor::isValidGeneIndex(int geneIndex) const
{
    return geneIndex >= 0 && geneIndex < toInt(_genome._genes.size());
}
