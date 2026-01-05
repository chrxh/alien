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
    phenotype.forEachCell([&](auto const& cell) {
        if (cell._geneIndex != startGeneIndex) {
            return;
        }
        cellIdsOnStartGene.insert(cell._id);
    });
    CHECK(!cellIdsOnStartGene.empty());
    auto firstCell = phenotype.getCellRef(getFirstElement(cellIdsOnStartGene));
    std::optional<CellDescription> secondCell;
    if (cellIdsOnStartGene.size() > 1) {
        secondCell = phenotype.getCellRef(getSecondElement(cellIdsOnStartGene));
    } else {
        // Only 1 cell with start gene? => try cells of referenced gene
        if (firstCell.getCellType() == CellType_Constructor) {
            auto refGeneIndex = std::get<ConstructorDescription>(firstCell._cellType)._geneIndex;
            phenotype.forEachCell([&](auto const& cell) {
                if (cell._geneIndex != refGeneIndex || cell._id == firstCell._id) {
                    return;
                }
                if (!secondCell.has_value() || cell._id > secondCell->_id) {
                    secondCell = cell;
                }
            });
        }
    }

    // Rotate preview
    if (secondCell.has_value()) {
        auto angle = Math::angleOfVector(secondCell->_pos - firstCell._pos);
        //if (!result.visualFrontAngle.has_value()) {
        result.visualFrontAngle = angle;
        //} else {
        //    result.visualFrontAngle.value() += Math::normalizedAngle(angle - result.visualFrontAngle.value(), -180.0f) / 3.0f;
        //    result.visualFrontAngle.value() = Math::normalizedAngle(result.visualFrontAngle.value(), 0.0);
        //}
        editService.rotate(phenotype, -result.visualFrontAngle.value());
    }

    // Create preview cells
    auto getNode = [&](CellDescription const& cell) -> NodeDescription const& { return genome._genes.at(cell._geneIndex)._nodes.at(cell._nodeIndex); };
    phenotype.forEachCell([&](CellDescription const& cell) {
        auto const& node = getNode(cell);
        auto const& color = cell._cellState == CellState_Ready ? node._color : -1;
        auto previewCell = CellPreviewDescription()
                               .id(cell._id)
                               .pos(cell._pos)
                               .color(color)
                               .geneIndex(cell._geneIndex)
                               .nodeIndex(cell._nodeIndex)
                               .signalState(cell._signalState);

        // Render as active if mode is Active or Conditional
        bool hasRestriction = (node._signalRestriction._mode == SignalRestrictionMode_Active || 
                               node._signalRestriction._mode == SignalRestrictionMode_Conditional) && 
                              !cell._connections.empty();
        if (hasRestriction) {
            auto otherCellId = cell._connections.front()._cellId;
            auto const& otherCell = phenotype.getCellRef(otherCellId, cache);
            auto baseAngle = Math::angleOfVector(otherCell._pos - cell._pos) + 180.0f + node._signalRestriction._baseAngle;
            auto signalAngleRestrictionStart = Math::getNormalizedAngle(baseAngle - node._signalRestriction._openingAngle / 2, 0);
            auto signalAngleRestrictionEnd = Math::getNormalizedAngle(baseAngle + node._signalRestriction._openingAngle / 2, 0);
            previewCell._signalRestriction = SignalRestrictionPreviewDescription().startAngle(signalAngleRestrictionStart).endAngle(signalAngleRestrictionEnd);
        }
        if (cell._signalState == SignalState_Active) {
            previewCell._signal = SignalPreviewDescription().channels(cell._signal._channels);
        }
        if (cell.getCellType() == CellType_Constructor) {
            if (!genome._genes.empty()) {
                auto nodeConstructor = std::get<ConstructorGenomeDescription>(node._cellType);
                previewCell._constructorGeneIndex = nodeConstructor._geneIndex;
            }
        }
        result.description._cells.emplace_back(previewCell);
    });

    // Determine arrow directions for each cell
    std::set<std::pair<uint64_t, uint64_t>> arrowFromCell1ToCell2;
    phenotype.forEachCell([&](CellDescription const& cell) {
        auto const& node = getNode(cell);
        auto signalAngleRestrictionStart = 180.0f + node._signalRestriction._baseAngle - node._signalRestriction._openingAngle / 2;
        auto signalAngleRestrictionEnd = 180.0f + node._signalRestriction._baseAngle + node._signalRestriction._openingAngle / 2;
        signalAngleRestrictionStart = Math::getNormalizedAngle(signalAngleRestrictionStart, 0.0f);
        signalAngleRestrictionEnd = Math::getNormalizedAngle(signalAngleRestrictionEnd, 0.0f);

        // For rendering, Active and Conditional modes are treated as having restriction
        bool hasNodeRestriction = (node._signalRestriction._mode == SignalRestrictionMode_Active || 
                                   node._signalRestriction._mode == SignalRestrictionMode_Conditional);

        auto summedAngle = 0.0f;
        for (int i = 0; i < cell._connections.size(); ++i) {
            if (i > 0) {
                summedAngle += cell._connections[i]._angleFromPrevious;
            }
            auto connectedCellId = cell._connections[i]._cellId;

            bool shouldAddArrow =
                !hasNodeRestriction || Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, summedAngle);

            if (shouldAddArrow) {
                arrowFromCell1ToCell2.insert({cell._id, connectedCellId});
            }
        }
    });

    // Create preview connections
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    phenotype.forEachCell([&](CellDescription const& cell) {
        for (const auto& connection : cell._connections) {
            uint64_t cellId1 = cell._id;
            uint64_t cellId2 = connection._cellId;

            auto connectionPair = std::make_pair(std::min(cellId1, cellId2), std::max(cellId1, cellId2));
            if (processedConnections.find(connectionPair) != processedConnections.end()) {
                continue;
            }
            processedConnections.insert(connectionPair);

            bool arrowToCell1 = arrowFromCell1ToCell2.contains({cellId2, cellId1});
            bool arrowToCell2 = arrowFromCell1ToCell2.contains({cellId1, cellId2});
            auto previewConnection = ConnectionPreviewDescription()
                                         .cell1(phenotype.getCellRef(cellId1, cache)._pos)
                                         .cell2(phenotype.getCellRef(cellId2, cache)._pos)
                                         .arrowToCell1(arrowToCell1)
                                         .arrowToCell2(arrowToCell2);
            result.description._connections.push_back(previewConnection);
        }
    });

    return result;
}
