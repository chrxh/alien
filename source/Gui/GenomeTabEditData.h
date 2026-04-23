#pragma once

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/ShapeGenerator.h>

#include "Definitions.h"

struct _GenomeTabEditData
{
    int id = 0;
    GenomeDesc genome;
    GenomeDesc origGenome;
    bool changesMade = false;  // true means diff between genome and origGenome

    std::optional<int> selectedGeneIndex;
    std::map<int, int> selectedNodeByGeneIndex;
    bool run = true;
    bool scheduleReload = false;
    int simulationSpeed = 50;  // In percent of full speed
    bool detailSimulation = false;
    bool showNodeIndex = true;  // true = show node index, false = show cell function

    std::optional<int> getSelectedNodeIndex() const
    {
        if (!selectedGeneIndex.has_value()) {
            return std::nullopt;
        }

        auto geneIndex = selectedGeneIndex.value();
        if (!selectedNodeByGeneIndex.contains(geneIndex)) {
            return std::nullopt;
        }

        return selectedNodeByGeneIndex.at(geneIndex);
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

    GeneDesc& getSelectedGeneRef() { return genome._genes.at(selectedGeneIndex.value()); }

    NodeDesc& getSelectedNodeRef()
    {
        auto& gene = getSelectedGeneRef();
        auto nodeIndex = getSelectedNodeIndex();
        return gene._nodes.at(nodeIndex.value());
    }

    void updateGeometry(ConstructorShape shape)
    {
        ShapeGenerator shapeGenerator;
        auto& gene = getSelectedGeneRef();
        auto numNodes = gene._nodes.size();
        int index = 0;
        for (auto& node : gene._nodes) {
            auto shapeGenerationResult = shapeGenerator.generateNextConstructionData(shape);
            if (index > 0 && index < numNodes - 1) {
                node._referenceAngle = shapeGenerationResult.angle;
            }
            ++index;
        }
    }
};
