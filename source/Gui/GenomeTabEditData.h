#pragma once

#include "EngineInterface/GenomeDescription.h"
#include "EngineInterface/ShapeGenerator.h"

#include "Definitions.h"

struct _GenomeTabEditData
{
    int id = 0;
    GenomeDescription genome;
    std::optional<int> selectedGeneIndex;
    std::map<int, int> selectedNodeByGeneIndex;

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

    GeneDescription& getSelectedGeneRef() { return genome._genes.at(selectedGeneIndex.value()); }

    NodeDescription& getSelectedNodeRef()
    {
        auto& gene = getSelectedGeneRef();
        auto nodeIndex = getSelectedNodeIndex();
        return gene._nodes.at(nodeIndex.value());
    }

    void updateGeometry(ConstructorShape shape)
    {
        auto shapeGenerator = ShapeGeneratorFactory::create(shape);
        if (!shapeGenerator) {
            return;
        }
        auto& gene = getSelectedGeneRef();
        gene._angleAlignment = shapeGenerator->getConstructorAngleAlignment();
        auto numNodes = gene._nodes.size();
        int index = 0;
        for (auto& node : gene._nodes) {
            auto shapeGenerationResult = shapeGenerator->generateNextConstructionData();
            if (index > 0 && index < numNodes - 1) {
                node._referenceAngle = shapeGenerationResult.angle;
            }
            if (index > 0) {
                node._numAdditionalConnections = shapeGenerationResult.numAdditionalConnections;
            }
            ++index;
        }
    }
};
