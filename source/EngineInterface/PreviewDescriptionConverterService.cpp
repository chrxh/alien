#include "PreviewDescriptionConverterService.h"

#include <set>

#include <Base/Math.h>

#include <EngineInterface/DescriptionEditService.h>

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

ConversionResult PreviewDescriptionConverterService::convertToPreviewDescription(
    GenomeDescription const& genome,
    int startGeneIndex,
    Description&& phenotype,
    std::optional<float> const& lastVisualFrontAngle) const
{
    ConversionResult result;
    result.frontAngle = genome._frontAngle;
    result.visualFrontAngle = lastVisualFrontAngle;

    auto const& editService = DescriptionEditService::get();

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
    std::optional<ObjectDescription> secondCell;
    if (cellIdsOnStartGene.size() > 1) {
        secondCell = phenotype.getObjectRef(getSecondElement(cellIdsOnStartGene));
    } else {
        // Only 1 cell with start gene? => try cells of referenced gene
        if (firstCell.getCellRef().getCellType() == CellType_Constructor) {
            auto refGeneIndex = std::get<ConstructorDescription>(firstCell.getCellRef()._cellType)._geneIndex;
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
    auto getNode = [&](ObjectDescription const& object) -> NodeDescription const& { return genome._genes.at(object.getCellRef()._geneIndex)._nodes.at(object.getCellRef()._nodeIndex); };
    for (auto const& object : phenotype._objects) {
        auto const& node = getNode(object);
        auto const& color = object.getCellRef()._cellState == CellState_Ready ? node._color : -1;
        auto previewCell = CellPreviewDescription()
                               .id(object._id)
                               .pos(object._pos)
                               .color(color)
                               .geneIndex(object.getCellRef()._geneIndex)
                               .nodeIndex(object.getCellRef()._nodeIndex)
                               .signalState(object.getCellRef()._signalState);

        // Render as active if mode is Active or Conditional
        bool hasRestriction = (node._signalRestriction._mode == SignalRestrictionMode_Active || 
                               node._signalRestriction._mode == SignalRestrictionMode_Conditional) && 
                              !object._connections.empty();
        if (hasRestriction) {
            auto otherCellId = object._connections.front()._objectId;
            auto const& otherCell = phenotype.getObjectRef(otherCellId, cache);
            auto baseAngle = Math::angleOfVector(otherCell._pos - object._pos) + 180.0f + node._signalRestriction._baseAngle;
            auto signalAngleRestrictionStart = Math::getNormalizedAngle(baseAngle - node._signalRestriction._openingAngle / 2, 0);
            auto signalAngleRestrictionEnd = Math::getNormalizedAngle(baseAngle + node._signalRestriction._openingAngle / 2, 0);
            previewCell._signalRestriction = SignalRestrictionPreviewDescription().startAngle(signalAngleRestrictionStart).endAngle(signalAngleRestrictionEnd);
        }
        if (object.getCellRef()._signalState == SignalState_Active) {
            previewCell._signal = SignalPreviewDescription().channels(object.getCellRef()._signal._channels);
        }
        if (object.getCellRef().getCellType() == CellType_Constructor) {
            if (!genome._genes.empty()) {
                auto nodeConstructor = std::get<ConstructorGenomeDescription>(node._cellType);
                previewCell._constructorGeneIndex = nodeConstructor._geneIndex;
            }
        }
        result.description._objects.emplace_back(previewCell);
    }

    // Determine arrow directions for each cell
    std::set<std::pair<uint64_t, uint64_t>> arrowFromCell1ToCell2;
    for (auto const& object : phenotype._objects) {
        auto const& node = getNode(object);
        auto signalAngleRestrictionStart = 180.0f + node._signalRestriction._baseAngle - node._signalRestriction._openingAngle / 2;
        auto signalAngleRestrictionEnd = 180.0f + node._signalRestriction._baseAngle + node._signalRestriction._openingAngle / 2;
        signalAngleRestrictionStart = Math::getNormalizedAngle(signalAngleRestrictionStart, 0.0f);
        signalAngleRestrictionEnd = Math::getNormalizedAngle(signalAngleRestrictionEnd, 0.0f);

        // For rendering, Active and Conditional modes are treated as having restriction
        bool hasNodeRestriction = (node._signalRestriction._mode == SignalRestrictionMode_Active || 
                                   node._signalRestriction._mode == SignalRestrictionMode_Conditional);

        auto summedAngle = 0.0f;
        for (int i = 0; i < object._connections.size(); ++i) {
            if (i > 0) {
                summedAngle += object._connections[i]._angleFromPrevious;
            }
            auto connectedCellId = object._connections[i]._objectId;

            bool shouldAddArrow =
                !hasNodeRestriction || Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, summedAngle);

            if (shouldAddArrow) {
                arrowFromCell1ToCell2.insert({object._id, connectedCellId});
            }
        }
    }

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

            bool arrowToObject1 = arrowFromCell1ToCell2.contains({objectId2, objectId1});
            bool arrowToObject2 = arrowFromCell1ToCell2.contains({objectId1, objectId2});
            auto previewConnection = ConnectionPreviewDescription()
                                         .object1(phenotype.getObjectRef(objectId1, cache)._pos)
                                         .object2(phenotype.getObjectRef(objectId2, cache)._pos)
                                         .arrowToObject1(arrowToObject1)
                                         .arrowToObject2(arrowToObject2);
            result.description._connections.push_back(previewConnection);
        }
    }

    return result;
}
