#pragma once

#include <optional>
#include <vector>

#include <Base/Singleton.h>

#include <Data/GenomeDesc.h>

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

enum class GenomeIssueSeverity
{
    Error,
    Warning
};

// A part of a genome that the simulation corrects in the genome of every offspring. Gene and node indices refer to the uncorrected genome.
struct GenomeIssue
{
    GenomeIssueType type = GenomeIssueType::GeneUnreachableFromRoot;
    int geneIndex = 0;
    std::optional<int> nodeIndex;
    std::vector<int> voidedNodeIndices;
    std::vector<int> relatedGeneIndices;  // The genes of a cycle, the genes keeping their separation or the removed gene
    bool removesGene = false;
    std::optional<int> causeIssueIndex;

    GenomeIssueSeverity getSeverity() const;
};

// Replays the corrections of MutationProcessor::correctGenome and GeneGraphProcessor in the same order
class GenomeValidationService
{
    MAKE_SINGLETON(GenomeValidationService);

public:
    std::vector<GenomeIssue> validate(GenomeDesc const& genome) const;

    // Unlike the simulation, a void boundary node becomes a base cell instead of a random cell type.
    // Returns the new index of every gene, -1 for a removed gene.
    std::vector<int> correct(GenomeDesc& genome) const;
};
