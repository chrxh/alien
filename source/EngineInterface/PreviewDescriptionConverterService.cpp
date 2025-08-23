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

    template <typename U, typename V>
    V getLastValue(std::map<U, V> const& m)
    {
        return (--m.end())->second;
    }

    template <typename U, typename V>
    V getSecondLastValue(std::map<U, V> const& m)
    {
        return (--(--m.end()))->second;
    }
}

auto PreviewDescriptionConverterService::convert(
    GenomeDescription const& genome,
    CollectionDescription&& phenotype,
    int startGeneIndex,
    std::optional<float> lastVisualFrontAngle) const -> ConversionResult
{
    ConversionResult result;
    result.visualFrontAngle = lastVisualFrontAngle;

    auto const& editService = DescriptionEditService::get();

    if (phenotype.isEmpty()) {
        return result;
    }
    auto cache = phenotype.createCache();

    // Center
    editService.setCenter(phenotype, {0.0f, 0.0f});

    // Get last constructed cell on principal gene
    std::map<int, std::map<int, std::set<uint64_t>>> geneAndNodeIndexToIds;  // Value has several ids in case of concantenations
    phenotype.forEachCell([&geneAndNodeIndexToIds](auto const& cell) { geneAndNodeIndexToIds[cell._geneIndex][cell._nodeIndex].insert(cell._id); });
    auto const& firstGene_NodeIndexToIds = geneAndNodeIndexToIds.at(startGeneIndex);
    auto const& lastConstructedCellIds = getLastValue(firstGene_NodeIndexToIds);
    auto const& lastConstructedCellId = getLastElement(lastConstructedCellIds);
    auto& lastConstructedCell = phenotype.getCellRef(lastConstructedCellId);


    // Try to get cell on principal gene with second last node index
    std::optional<uint64_t> prevLastConstructedCellId;
    if (firstGene_NodeIndexToIds.size() > 1) {
        auto const& secondLastConstructedCellIds = getSecondLastValue(firstGene_NodeIndexToIds);
        prevLastConstructedCellId = getLastElement(secondLastConstructedCellIds);
    }

    // Try to get cell on principal gene on second last concatenation 
    else if (lastConstructedCellIds.size() > 1) {
        prevLastConstructedCellId = getSecondLastElement(lastConstructedCellIds);
    }

    // Try to get cell which is constructed of last cell on principal gene
    else if (lastConstructedCell.getCellType() == CellType_Constructor) {
        auto const& constructor = std::get<ConstructorDescription>(lastConstructedCell._cellTypeData);
        auto geneIndex = constructor._geneIndex;
        if (geneAndNodeIndexToIds.contains(geneIndex)) {
            auto& secondGene_NodeIndexToIds = geneAndNodeIndexToIds.at(geneIndex);
            prevLastConstructedCellId = getLastElement(getLastValue(secondGene_NodeIndexToIds));
        }
    }

    if (prevLastConstructedCellId.has_value()) {
        auto& prevLastConstructedCell = phenotype.getCellRef(prevLastConstructedCellId.value());

        auto angle = Math::angleOfVector(prevLastConstructedCell._pos - lastConstructedCell._pos);
        if (!result.visualFrontAngle.has_value()) {
            result.visualFrontAngle = angle;
        } else {
            result.visualFrontAngle.value() += Math::normalizedAngle(angle - result.visualFrontAngle.value(), -180.0f) / 5.0f;
            result.visualFrontAngle.value() = Math::normalizedAngle(result.visualFrontAngle.value(), 0.0);
        }
        editService.rotate(phenotype, -result.visualFrontAngle.value());
    }

    // Create preview cells
    auto getNode = [&](CellDescription const& cell) -> NodeDescription const& { return genome._genes.at(cell._geneIndex)._nodes.at(cell._nodeIndex); };
    //auto getGene = [&](CellDescription const& cell) -> GeneDescription const& { return genome._genes.at(cell._geneIndex); };
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
