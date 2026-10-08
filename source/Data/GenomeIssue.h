#pragma once

#include <optional>
#include <set>
#include <vector>

enum class GenomeIssueType
{
    VoidBoundaryNode,
    NodesCutOffFromLastNode,
    CycleAvoidingRootGene,
    GeneUnreachableFromRoot,
    TooManyGenesWithSeparation,
    ConstructsRemovedGene,
    InjectsRemovedGene
};

struct GenomeIssue
{
    GenomeIssueType type = GenomeIssueType::GeneUnreachableFromRoot;
    int geneIndex = 0;
    std::optional<int> nodeIndex;
    std::vector<int> cutOffNodeIndices;
    std::vector<int> cycleGeneIndices;
    std::vector<int> geneIndicesKeepingSeparation;
    std::optional<int> referencedGeneIndex;
    bool removesGene = false;
    std::optional<int> causeIssueIndex;

    bool isFollowUp() const { return causeIssueIndex.has_value(); }

    static std::vector<GenomeIssue> filterByGene(std::vector<GenomeIssue> const& issues, int geneIndex);
    static std::set<int> getRemovedGeneIndices(std::vector<GenomeIssue> const& issues);
};
