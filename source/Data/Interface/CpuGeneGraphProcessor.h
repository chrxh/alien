#pragma once

#include <map>
#include <optional>
#include <vector>

#include "GenomeDesc.h"
#include "GenomeIssue.h"

class CpuGeneGraphProcessor
{
public:
    explicit CpuGeneGraphProcessor(GenomeDesc const& genome);

    std::vector<GenomeIssue> process();

private:
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

    static std::vector<int> calcNodesCutOffFromLastNode(GeneState const& gene);

    void correctVoidBoundaryNodes();
    void voidNodesCutOffFromLastNode();
    void removeCyclesAvoidingRootGene();
    void removeGenesUnreachableFromRoot();
    void limitGenesWithSeparation();

    void removeGenes(std::map<int, int> const& removalIssueIndexByGeneIndex);
    std::optional<int> findCauseOfUnreachability(int geneIndex) const;
    std::vector<int> getDisconnectedGeneIndices(GenomeIssue const& issue) const;
    bool isChangedByVoiding(int geneIndex, int nodeIndex) const;
    std::vector<int> getGenePositions() const;
    bool isValidGeneIndex(int geneIndex) const;

    GenomeDesc const& _genome;
    bool _correctsVoidBoundaryNodes = false;
    std::vector<GeneState> _genes;
    std::vector<GenomeIssue> _issues;
};
