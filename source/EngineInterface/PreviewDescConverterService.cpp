#include "PreviewDescConverterService.h"

#include <set>

#include <Base/Math.h>

#include <EngineInterface/DescEditService.h>

#include "SpaceCalculator.h"

namespace
{
    template <typename U>
    U getFirstElement(std::set<U> const& s)
    {
        //return *(--s.end());
        return *s.begin();
    }

    template <typename U>
    U getSecondElement(std::set<U> const& s)
    {
        //return *(--(-- s.end()));
        return *(++s.begin());
    }
}

ConversionResult PreviewDescConverterService::convertToPreviewDesc(
    GenomeDesc const& genome,
    int startGeneIndex,
    Desc&& phenotype,
    std::optional<float> const& lastVisualFrontAngle) const
{
    ConversionResult result;
    result.frontAngle = genome._frontAngle;
    result.visualFrontAngle = lastVisualFrontAngle;

    auto const& editService = DescEditService::get();

    if (phenotype.isEmpty()) {
        return result;
    }
    CHECK(phenotype._creatures.size() == 1);

    editService.flattenTopology(phenotype, {PREVIEW_WIDTH, PREVIEW_HEIGHT});

    auto cache = phenotype.createCache();

    // Center
    editService.setCenter(phenotype, {0, 0});

    // Get first and second constructed cell on start gene
    std::set<uint64_t> cellIdsOnStartGene;
    for (auto const& object : phenotype._objects) {
        if (object.getCellRef()._geneIndex != startGeneIndex) {
            continue;
        }
        cellIdsOnStartGene.insert(object._id);
    }
    CHECK(!cellIdsOnStartGene.empty());
    auto firstCell = phenotype.getObjectRef(getFirstElement(cellIdsOnStartGene));
    std::optional<ObjectDesc> secondCell;
    if (cellIdsOnStartGene.size() > 1) {
        secondCell = phenotype.getObjectRef(getSecondElement(cellIdsOnStartGene));
    } else {
        // Only 1 cell with start gene? => try cells of referenced gene
        if (firstCell.getCellRef()._constructor.has_value()) {
            auto refGeneIndex = firstCell.getCellRef()._constructor.value()._geneIndex;
            for (auto const& object : phenotype._objects) {
                if (object.getCellRef()._geneIndex != refGeneIndex || object._id == firstCell._id) {
                    continue;
                }
                if (!secondCell.has_value() || object._id > secondCell->_id) {
                    secondCell = object;
                }
            }
        }
    }

    // Rotate preview
    if (secondCell.has_value()) {
        auto angle = Math::angleOfVector(secondCell->_pos - firstCell._pos);
        result.visualFrontAngle = angle;
        editService.rotate(phenotype, -result.visualFrontAngle.value());
    }

    // Create preview cells
    auto getNode = [&](ObjectDesc const& object) -> NodeDesc const& {
        return genome._genes.at(object.getCellRef()._geneIndex)._nodes.at(object.getCellRef()._nodeIndex);
    };
    for (auto const& object : phenotype._objects) {
        auto const& node = getNode(object);
        auto const& color = object.getCellRef()._cellState == CellState_Ready ? node._color : -1;
        auto previewCell =
            CellPreviewDesc().id(object._id).pos(object._pos).color(color).geneIndex(object.getCellRef()._geneIndex).nodeIndex(object.getCellRef()._nodeIndex);

        previewCell._signal = SignalPreviewDesc().channels(object.getCellRef()._signal._channels);
        if (node._constructor.has_value()) {
            if (!genome._genes.empty()) {
                auto nodeConstructor = node._constructor.value();
                previewCell._constructorGeneIndex = nodeConstructor._geneIndex;
            }
        }
        result.description._objects.emplace_back(previewCell);
    }

    // Helper to find the connection weight from genome for a given connection
    auto getConnectionWeight = [&getNode](ObjectDesc const& sourceObject, uint64_t targetId) -> float {
        for (int i = 0, size = sourceObject._connections.size(); i < size; ++i) {
            if (sourceObject._connections.at(i)._objectId == targetId) {
                auto const& cw = getNode(sourceObject)._neuralNetwork._connectionWeights;
                return cw.at(i);
            }
        }
        return 0.0f;
    };

    // Create preview connections
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    for (auto const& object : phenotype._objects) {
        for (const auto& connection : object._connections) {
            uint64_t objectId1 = object._id;
            uint64_t objectId2 = connection._objectId;

            auto connectionPair = std::make_pair(std::min(objectId1, objectId2), std::max(objectId1, objectId2));
            if (processedConnections.find(connectionPair) != processedConnections.end()) {
                continue;
            }
            processedConnections.insert(connectionPair);

            auto const& object1 = phenotype.getObjectRef(objectId1, cache);
            auto const& object2 = phenotype.getObjectRef(objectId2, cache);

            auto previewConnection = ConnectionPreviewDesc()
                                         .object1(object1._pos)
                                         .object2(object2._pos)
                                         .connectionWeightToObject1(getConnectionWeight(object1, objectId2))
                                         .connectionWeightToObject2(getConnectionWeight(object2, objectId1));
            result.description._connections.push_back(previewConnection);
        }
    }

    return result;
}
