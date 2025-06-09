#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "Definitions.h"

struct _CreatureTabEditData
{
    GenomeDescription_New genome;
    std::optional<int> selectedGene;
    std::map<int, int> selectedNodeByGeneIndex;

    std::optional<int> getSelectedNode() const
    {
        if (!selectedGene.has_value()) {
            return std::nullopt;
        }

        auto geneIndex = selectedGene.value();
        if (!selectedNodeByGeneIndex.contains(geneIndex)) {
            return std::nullopt;
        }

        return selectedNodeByGeneIndex.at(geneIndex);
    }

    void setSelectedNode(int value)
    {
        if (!selectedGene.has_value()) {
            return;
        }

        auto geneIndex = selectedGene.value();
        selectedNodeByGeneIndex.insert_or_assign(geneIndex, value);
    }
};
