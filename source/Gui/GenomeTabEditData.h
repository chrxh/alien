#pragma once

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/ShapeGenerator.h>

#include "Definitions.h"

// Determines which of the two levels the inspector shows: a gene row or a node row was selected last
enum class GenomeSelectionLevel
{
    Gene,
    Node
};

struct _GenomeTabEditData
{
    int id = 0;
    GenomeDesc genome;
    GenomeDesc origGenome;
    bool changesMade = false;  // true means diff between genome and origGenome

    std::optional<int> selectedGeneIndex;
    std::map<int, int> selectedNodeByGeneIndex;
    GenomeSelectionLevel selectionLevel = GenomeSelectionLevel::Gene;
    bool run = true;
    bool scheduleReload = false;
    int simulationSpeed = 50;  // In percent of full speed
    bool detailSimulation = false;

    bool hasValidGeneIndex(int geneIndex) const { return geneIndex >= 0 && static_cast<size_t>(geneIndex) < genome._genes.size(); }

    bool hasValidNodeIndex(int geneIndex, int nodeIndex) const
    {
        return hasValidGeneIndex(geneIndex) && nodeIndex >= 0 && static_cast<size_t>(nodeIndex) < genome._genes.at(geneIndex)._nodes.size();
    }

    std::optional<int> getSelectedNodeIndex() const
    {
        if (!selectedGeneIndex.has_value()) {
            return std::nullopt;
        }

        auto geneIndex = selectedGeneIndex.value();
        if (!hasValidGeneIndex(geneIndex) || !selectedNodeByGeneIndex.contains(geneIndex)) {
            return std::nullopt;
        }

        auto nodeIndex = selectedNodeByGeneIndex.at(geneIndex);
        if (!hasValidNodeIndex(geneIndex, nodeIndex)) {
            return std::nullopt;
        }

        return nodeIndex;
    }

    void setSelectedNodeIndex(std::optional<int> value)
    {
        if (!selectedGeneIndex.has_value()) {
            return;
        }

        auto geneIndex = selectedGeneIndex.value();
        if (value.has_value()) {
            selectedNodeByGeneIndex.insert_or_assign(geneIndex, value.value());
        } else {
            selectedNodeByGeneIndex.erase(geneIndex);
        }
    }

    // True if the inspector should show the node level: a node row is selected and still valid
    bool isNodeLevelSelected() const { return selectionLevel == GenomeSelectionLevel::Node && getSelectedNodeIndex().has_value(); }

    void selectGene(int geneIndex)
    {
        selectedGeneIndex = geneIndex;
        selectionLevel = GenomeSelectionLevel::Gene;
    }

    void selectNode(int geneIndex, int nodeIndex)
    {
        selectedGeneIndex = geneIndex;
        setSelectedNodeIndex(nodeIndex);
        selectionLevel = GenomeSelectionLevel::Node;
    }

    GeneDesc& getSelectedGeneRef() { return genome._genes.at(selectedGeneIndex.value()); }

    NodeDesc& getSelectedNodeRef()
    {
        auto& gene = getSelectedGeneRef();
        auto nodeIndex = getSelectedNodeIndex();
        return gene._nodes.at(nodeIndex.value());
    }

    // The reference angles of the middle nodes are derived data: the constructor regenerates them
    // from the gene's shape during construction
    static void updateGeneGeometry(GeneDesc& gene)
    {
        ShapeGenerator shapeGenerator;
        auto numNodes = gene._nodes.size();
        int index = 0;
        for (auto& node : gene._nodes) {
            auto shapeGenerationResult = shapeGenerator.generateNextConstructionData(gene._shape);
            if (index > 0 && index < numNodes - 1) {
                node._referenceAngle = shapeGenerationResult.angle;
            }
            ++index;
        }
    }

    void updateGeometry() { updateGeneGeometry(getSelectedGeneRef()); }
};
