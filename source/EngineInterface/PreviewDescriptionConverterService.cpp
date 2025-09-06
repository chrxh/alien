#include "PreviewDescriptionConverterService.h"

#include <set>

#include "Base/Math.h"

#include "EngineInterface/DescriptionEditService.h"

namespace
{
    template <typename U>
    U getLastElement(std::set<U> const& s)
    {
        return *(--s.end());
    }

    template <typename U>
    U getSecondLastElement(std::set<U> const& s)
    {
        return *(--(--s.end()));
    }
}

ConversionResult PreviewDescriptionConverterService::convert(
    GenomeDescription const& genome,
    CollectionDescription&& phenotype,
    int startGeneIndex,
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
    auto cache = phenotype.createCache();

    // Center
    editService.setCenter(phenotype, {0, 0});

    // Get last and second last constructed cell on start gene
    std::set<uint64_t> cellIdsOnStartGene;
    phenotype.forEachCell([&](auto const& cell) {
        if (cell._geneIndex != startGeneIndex) {
            return;
        }
        cellIdsOnStartGene.insert(cell._id);
    });
    CHECK(!cellIdsOnStartGene.empty());
    auto lastCell = phenotype.getCellRef(getLastElement(cellIdsOnStartGene));
    std::optional<CellDescription> secondLastCell;
    if (cellIdsOnStartGene.size() > 1) {
        secondLastCell = phenotype.getCellRef(getSecondLastElement(cellIdsOnStartGene));
    } else {
        // Only 1 cell with start gene? => try cells of referenced gene
        if (lastCell.getCellType() == CellType_Constructor) {
            auto refGeneIndex = std::get<ConstructorDescription>(lastCell._cellTypeData)._geneIndex;
            phenotype.forEachCell([&](auto const& cell) {
                if (cell._geneIndex != refGeneIndex || cell._id == lastCell._id) {
                    return;
                }
                if (!secondLastCell.has_value() || cell._id > secondLastCell->_id) {
                    secondLastCell = cell;
                }
            });
        }
    }

    // Rotate preview
    if (secondLastCell.has_value()) {
        auto angle = Math::angleOfVector(secondLastCell->_pos - lastCell._pos);
        if (!result.visualFrontAngle.has_value()) {
            result.visualFrontAngle = angle;
        } else {
            result.visualFrontAngle.value() += Math::normalizedAngle(angle - result.visualFrontAngle.value(), -180.0f) / 10.0f;
            result.visualFrontAngle.value() = Math::normalizedAngle(result.visualFrontAngle.value(), 0.0);
        }
        editService.rotate(phenotype, -result.visualFrontAngle.value());
    }

    // Create preview cells
    auto getNode = [&](CellDescription const& cell) -> NodeDescription const& { return genome._genes.at(cell._geneIndex)._nodes.at(cell._nodeIndex); };
    phenotype.forEachCell([&](CellDescription const& cell) {
        auto const& node = getNode(cell);
        auto const& color = node._color;
        auto previewCell = CellPreviewDescription().pos(cell._pos).color(color).geneIndex(cell._geneIndex).nodeIndex(cell._nodeIndex);

        if (node._signalRestriction._active && !cell._connections.empty()) {
            auto otherCellId = cell._connections.front()._cellId;
            auto const& otherCell = phenotype.getCellRef(otherCellId, cache);
            auto baseAngle = Math::angleOfVector(otherCell._pos - cell._pos) + 180.0f + node._signalRestriction._baseAngle;
            auto signalAngleRestrictionStart = Math::normalizedAngle(baseAngle - node._signalRestriction._openingAngle / 2, 0);
            auto signalAngleRestrictionEnd = Math::normalizedAngle(baseAngle + node._signalRestriction._openingAngle / 2, 0);
            previewCell._signalRestriction = SignalRestrictionPreviewDescription().startAngle(signalAngleRestrictionStart).endAngle(signalAngleRestrictionEnd);
        }
        if (cell.getCellType() == CellType_Constructor) {
            auto constructor = std::get<ConstructorDescription>(cell._cellTypeData);
            if (!genome._genes.empty()) {
                auto nodeConstructor = std::get<ConstructorGenomeDescription>(node._cellTypeData);
                if (constructor._geneIndex == genome._genes.size()) {
                    previewCell._constructorGeneIndex = nodeConstructor._geneIndex;
                }
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
        signalAngleRestrictionStart = Math::normalizedAngle(signalAngleRestrictionStart, 0.0f);
        signalAngleRestrictionEnd = Math::normalizedAngle(signalAngleRestrictionEnd, 0.0f);

        auto summedAngle = 0.0f;
        for (int i = 0; i < cell._connections.size(); ++i) {
            if (i > 0) {
                summedAngle += cell._connections[i]._angleFromPrevious;
            }
            auto connectedCellId = cell._connections[i]._cellId;

            bool shouldAddArrow =
                !node._signalRestriction._active || Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, summedAngle);

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
