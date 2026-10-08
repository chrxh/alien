#include "GenomeValidationService.h"

#include <algorithm>
#include <map>
#include <ranges>
#include <set>
#include <numeric>
#include <span>

#include <boost/range/adaptor/indexed.hpp>

#include <Base/Definitions.h>

#include "ShapeGenerator.h"

namespace
{
    auto constexpr MaxGenesWithSeparation = 2;
}

GenomeIssueSeverity GenomeIssue::getSeverity() const
{
    return removesGene || !voidedNodeIndices.empty() ? GenomeIssueSeverity::Error : GenomeIssueSeverity::Warning;
}

namespace
{
    struct NodeState
    {
        bool isVoid = false;
        std::optional<int> constructorGeneIndex;
        bool separation = false;
        std::optional<int> injectorGeneIndex;
    };

    struct GeneState
    {
        int geneIndex = 0;
        ConstructorShape shape = ConstructorShape_Segment;
        bool homogeneousCellType = false;
        std::vector<NodeState> nodes;
    };

    struct SearchStackEntry
    {
        int geneIndex = 0;
        int nodeIndex = 0;
    };

    // Mirrors GeneGraphProcessor::voidUnreachableNodes: the union-find components of the node graph decide which nodes stay connected to the last node
    std::vector<int> calcNodesCutOffFromLastNode(GeneState const& gene)
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

    // Replays the corrections on a reduced copy of the genome in which genes keep their original index
    class CorrectionReplay
    {
    public:
        explicit CorrectionReplay(GenomeDesc const& genome);

        std::vector<GenomeIssue> run();

    private:
        void correctBoundaryNodes();
        void voidNodesCutOffFromLastNode();
        void removeCyclesAvoidingRootGene();
        void removeGenesUnreachableFromRoot();
        void limitGenesWithSeparation();

        void removeGenes(std::map<int, int> const& removalIssueByGeneIndex);
        std::optional<int> findCauseOfUnreachability(int geneIndex) const;
        std::vector<int> getDisconnectedGeneIndices(GenomeIssue const& issue) const;
        bool isOriginallyReachable(int fromGeneIndex, int toGeneIndex) const;
        std::vector<int> getGenePositions() const;
        bool isValidGeneIndex(int geneIndex) const;

        GenomeDesc const& _genome;
        bool _correctsBoundaryNodes = false;
        std::vector<GeneState> _genes;
        std::vector<GenomeIssue> _issues;
    };

    CorrectionReplay::CorrectionReplay(GenomeDesc const& genome)
        : _genome(genome)
    {
        // MutationProcessor::correctGenome only follows structural node mutations; otherwise the gene graph corrections handle void boundary nodes
        auto const& rates = genome._mutationRates;
        _correctsBoundaryNodes = rates._extendGeneMutation._geneProbability > 0 || rates._addNodeMutation._nodeProbability > 0
            || rates._trimGeneMutation._geneProbability > 0 || rates._deleteNodeMutation._nodeProbability > 0
            || rates._copyNodeSectionMutation._geneProbability > 0 || rates._moveNodeSectionMutation._geneProbability > 0;

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

    std::vector<GenomeIssue> CorrectionReplay::run()
    {
        if (_genes.empty()) {
            return {};
        }
        if (_correctsBoundaryNodes) {
            correctBoundaryNodes();
        }
        voidNodesCutOffFromLastNode();
        removeCyclesAvoidingRootGene();
        removeGenesUnreachableFromRoot();
        limitGenesWithSeparation();
        return _issues;
    }

    void CorrectionReplay::correctBoundaryNodes()
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

                // With a homogeneous cell type the last node is expressed with the cell type of the first node anyway
                if (nodeIndex == 0 || !gene.homogeneousCellType) {
                    _issues.emplace_back(GenomeIssue{.type = GenomeIssueType::VoidBoundaryNode, .geneIndex = gene.geneIndex, .nodeIndex = nodeIndex});
                }
            }
        }
    }

    void CorrectionReplay::voidNodesCutOffFromLastNode()
    {
        std::vector<std::vector<int>> cutOffNodeIndicesByGene;
        for (auto const& gene : _genes) {
            cutOffNodeIndicesByGene.emplace_back(calcNodesCutOffFromLastNode(gene));
        }

        // The genome never becomes empty: if every gene would be removed, the first one is kept
        auto losesFirstNode = [](std::vector<int> const& cutOffNodeIndices) { return !cutOffNodeIndices.empty() && cutOffNodeIndices.front() == 0; };
        auto keepsFirstGene = std::ranges::all_of(cutOffNodeIndicesByGene, losesFirstNode);

        std::map<int, int> removalIssueByGeneIndex;
        for (auto&& [gene, cutOffNodeIndices] : std::views::zip(_genes, cutOffNodeIndicesByGene)) {
            if (cutOffNodeIndices.empty()) {
                continue;
            }
            GenomeIssue issue{.type = GenomeIssueType::NodesCutOffFromLastNode, .geneIndex = gene.geneIndex};
            for (auto nodeIndex : cutOffNodeIndices) {
                auto& node = gene.nodes.at(nodeIndex);
                if (!node.isVoid) {
                    issue.voidedNodeIndices.emplace_back(nodeIndex);
                }
                node = NodeState{.isVoid = true};
            }
            issue.removesGene = losesFirstNode(cutOffNodeIndices) && !(keepsFirstGene && gene.geneIndex == _genes.front().geneIndex);
            if (issue.voidedNodeIndices.empty() && !issue.removesGene) {
                continue;
            }

            if (!_correctsBoundaryNodes) {
                auto const& nodes = _genome._genes.at(gene.geneIndex)._nodes;
                if (nodes.front().getCellType() == CellType_Void) {
                    issue.nodeIndex = 0;
                } else if (nodes.back().getCellType() == CellType_Void) {
                    issue.nodeIndex = toInt(nodes.size()) - 1;
                }
            }
            if (issue.removesGene) {
                removalIssueByGeneIndex.emplace(gene.geneIndex, toInt(_issues.size()));
            }
            _issues.emplace_back(issue);
        }
        removeGenes(removalIssueByGeneIndex);
    }

    void CorrectionReplay::removeCyclesAvoidingRootGene()
    {
        // Mirrors GeneGraphProcessor::removeCyclesNotThroughRoot: a depth-first search over all genes except the root gene turns off
        // every constructor that leads back to a gene on the search stack
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
                        issue.relatedGeneIndices.emplace_back(entry.geneIndex);
                    }
                    issue.relatedGeneIndices.emplace_back(targetGeneIndex);
                    _issues.emplace_back(issue);
                    node.constructorGeneIndex.reset();
                } else if (states.at(targetGeneIndex) == SearchState::NotVisited) {
                    states.at(targetGeneIndex) = SearchState::OnStack;
                    stack.emplace_back(SearchStackEntry{.geneIndex = targetGeneIndex});
                }
            }
        }
    }

    void CorrectionReplay::removeGenesUnreachableFromRoot()
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

        std::map<int, int> removalIssueByGeneIndex;
        for (auto const& gene : _genes) {
            if (reachedGeneIndices.contains(gene.geneIndex)) {
                continue;
            }
            auto issue = GenomeIssue{
                .type = GenomeIssueType::GeneUnreachableFromRoot,
                .geneIndex = gene.geneIndex,
                .removesGene = true,
                .causeIssueIndex = findCauseOfUnreachability(gene.geneIndex)};
            removalIssueByGeneIndex.emplace(gene.geneIndex, toInt(_issues.size()));
            _issues.emplace_back(issue);
        }
        removeGenes(removalIssueByGeneIndex);
    }

    void CorrectionReplay::limitGenesWithSeparation()
    {
        // Mirrors GeneGraphProcessor::limitGenesWithSeparation: a depth-first search from the root gene meets the constructors in construction order
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
                if (toInt(genesWithSeparation.size()) < MaxGenesWithSeparation) {
                    genesWithSeparation.emplace_back(targetGeneIndex);
                } else {
                    _issues.emplace_back(GenomeIssue{
                        .type = GenomeIssueType::TooManyGenesWithSeparation,
                        .geneIndex = geneIndex,
                        .nodeIndex = nodeIndex,
                        .relatedGeneIndices = genesWithSeparation});
                    node.separation = false;
                }
            }
            if (visitedGeneIndices.insert(targetGeneIndex).second) {
                stack.emplace_back(SearchStackEntry{.geneIndex = targetGeneIndex});
            }
        }
    }

    void CorrectionReplay::removeGenes(std::map<int, int> const& removalIssueByGeneIndex)
    {
        if (removalIssueByGeneIndex.empty()) {
            return;
        }
        std::erase_if(_genes, [&](GeneState const& gene) { return removalIssueByGeneIndex.contains(gene.geneIndex); });

        // As in GeneGraphProcessor::removeMarkedGenes, a constructor of a removed gene is turned off and an injector falls back to the first gene
        for (auto& gene : _genes) {
            for (auto nodeIndex : std::views::iota(0, toInt(gene.nodes.size()))) {
                auto& node = gene.nodes.at(nodeIndex);
                if (node.constructorGeneIndex.has_value() && removalIssueByGeneIndex.contains(node.constructorGeneIndex.value())) {
                    _issues.emplace_back(GenomeIssue{
                        .type = GenomeIssueType::ConstructsRemovedGene,
                        .geneIndex = gene.geneIndex,
                        .nodeIndex = nodeIndex,
                        .relatedGeneIndices = {node.constructorGeneIndex.value()},
                        .causeIssueIndex = removalIssueByGeneIndex.at(node.constructorGeneIndex.value())});
                    node.constructorGeneIndex.reset();
                }
                if (node.injectorGeneIndex.has_value() && removalIssueByGeneIndex.contains(node.injectorGeneIndex.value())) {
                    _issues.emplace_back(GenomeIssue{
                        .type = GenomeIssueType::InjectsRemovedGene,
                        .geneIndex = gene.geneIndex,
                        .nodeIndex = nodeIndex,
                        .relatedGeneIndices = {node.injectorGeneIndex.value()},
                        .causeIssueIndex = removalIssueByGeneIndex.at(node.injectorGeneIndex.value())});
                    node.injectorGeneIndex = _genes.front().geneIndex;
                }
            }
        }
    }

    // A gene that was reachable before is attributed to the first earlier correction that disconnected a constructor leading to it
    std::optional<int> CorrectionReplay::findCauseOfUnreachability(int geneIndex) const
    {
        if (!isOriginallyReachable(0, geneIndex)) {
            return std::nullopt;
        }
        for (auto const& [issueIndex, issue] : _issues | boost::adaptors::indexed(0)) {
            for (auto disconnectedGeneIndex : getDisconnectedGeneIndices(issue)) {
                if (isOriginallyReachable(disconnectedGeneIndex, geneIndex)) {
                    return toInt(issueIndex);
                }
            }
        }
        return std::nullopt;
    }

    std::vector<int> CorrectionReplay::getDisconnectedGeneIndices(GenomeIssue const& issue) const
    {
        auto const& nodes = _genome._genes.at(issue.geneIndex)._nodes;
        std::vector<int> nodeIndices;
        if (issue.removesGene) {
            nodeIndices.resize(nodes.size());
            std::iota(nodeIndices.begin(), nodeIndices.end(), 0);
        } else if (issue.type == GenomeIssueType::NodesCutOffFromLastNode) {
            nodeIndices = issue.voidedNodeIndices;
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

    bool CorrectionReplay::isOriginallyReachable(int fromGeneIndex, int toGeneIndex) const
    {
        std::set<int> reachedGeneIndices = {fromGeneIndex};
        std::vector<int> genesToScan = {fromGeneIndex};
        while (!genesToScan.empty()) {
            auto geneIndex = genesToScan.back();
            genesToScan.pop_back();
            if (geneIndex == toGeneIndex) {
                return true;
            }
            for (auto const& node : _genome._genes.at(geneIndex)._nodes) {
                if (node._constructor.has_value() && isValidGeneIndex(node._constructor->_geneIndex)
                    && reachedGeneIndices.insert(node._constructor->_geneIndex).second) {
                    genesToScan.emplace_back(node._constructor->_geneIndex);
                }
            }
        }
        return false;
    }

    std::vector<int> CorrectionReplay::getGenePositions() const
    {
        std::vector<int> result(_genome._genes.size(), -1);
        for (auto const& [position, gene] : _genes | boost::adaptors::indexed(0)) {
            result.at(gene.geneIndex) = toInt(position);
        }
        return result;
    }

    bool CorrectionReplay::isValidGeneIndex(int geneIndex) const
    {
        return geneIndex >= 0 && geneIndex < toInt(_genome._genes.size());
    }
}

std::vector<GenomeIssue> GenomeValidationService::validate(GenomeDesc const& genome) const
{
    return CorrectionReplay(genome).run();
}

std::vector<int> GenomeValidationService::correct(GenomeDesc& genome) const
{
    std::set<int> removedGeneIndices;
    for (auto const& issue : validate(genome)) {
        auto& gene = genome._genes.at(issue.geneIndex);
        if (issue.removesGene) {
            removedGeneIndices.insert(issue.geneIndex);
        }
        for (auto nodeIndex : issue.voidedNodeIndices) {
            auto& node = gene._nodes.at(nodeIndex);
            node._cellType = VoidGenomeDesc();
            node._constructor.reset();
        }
        if (!issue.nodeIndex.has_value()) {
            continue;
        }
        auto& node = gene._nodes.at(issue.nodeIndex.value());
        if (issue.type == GenomeIssueType::VoidBoundaryNode) {
            node._cellType = BaseGenomeDesc();
        } else if (issue.type == GenomeIssueType::CycleAvoidingRootGene || issue.type == GenomeIssueType::ConstructsRemovedGene) {
            node._constructor.reset();
        } else if (issue.type == GenomeIssueType::TooManyGenesWithSeparation) {
            node._constructor->_separation = false;
        }
    }

    std::vector<int> result;
    auto numRemainingGenes = 0;
    for (auto geneIndex : std::views::iota(0, toInt(genome._genes.size()))) {
        result.emplace_back(removedGeneIndices.contains(geneIndex) ? -1 : numRemainingGenes++);
    }

    auto isValidGeneIndex = [&](int geneIndex) { return geneIndex >= 0 && geneIndex < toInt(result.size()); };
    for (auto& gene : genome._genes) {
        for (auto& node : gene._nodes) {
            if (node._constructor.has_value() && isValidGeneIndex(node._constructor->_geneIndex)) {
                auto newGeneIndex = result.at(node._constructor->_geneIndex);
                if (newGeneIndex >= 0) {
                    node._constructor->_geneIndex = newGeneIndex;
                } else {
                    node._constructor.reset();
                }
            }
            if (auto injector = std::get_if<InjectorGenomeDesc>(&node._cellType); injector != nullptr && isValidGeneIndex(injector->_geneIndex)) {
                injector->_geneIndex = std::max(0, result.at(injector->_geneIndex));
            }
        }
    }

    std::vector<GeneDesc> remainingGenes;
    for (auto&& [gene, newGeneIndex] : std::views::zip(genome._genes, result)) {
        if (newGeneIndex >= 0) {
            remainingGenes.emplace_back(std::move(gene));
        }
    }
    genome._genes = std::move(remainingGenes);
    return result;
}
