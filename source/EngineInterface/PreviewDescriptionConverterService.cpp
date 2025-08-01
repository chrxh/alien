#include "PreviewDescriptionConverterService.h"

#include <set>
#include <map>

#include "Base/Math.h"

#include "EngineInterface/DescriptionEditService.h"

PreviewDescription PreviewDescriptionConverterService::convert(GenomeDescription const& genome, CollectionDescription&& phenotype) const
{
    PreviewDescription result;

    auto const& editService = DescriptionEditService::get();

    // Remove seed
    uint64_t smallestCellId = 0xffffffffffffffff;
    phenotype.forEachCell([&smallestCellId](auto const& cell) { smallestCellId = std::min(smallestCellId, cell._id); });
    editService.removeCell(phenotype, smallestCellId);
    auto cache = phenotype.createCache();
    if (phenotype.isEmpty()) {
        return result;
    }

    // Center
    editService.setCenter(phenotype, {0.0f, 0.0f});

    // Try to get last and previous last constructed cell on principal gene
    std::map<int, std::map<int, uint64_t>> geneAndNodeIndexToId;
    phenotype.forEachCell([&geneAndNodeIndexToId](auto const& cell) { geneAndNodeIndexToId[cell._geneIndex][cell._nodeIndex] = cell._id; });

    auto& firstGene_NodeIndexToId = geneAndNodeIndexToId.at(0);
    auto lastConstructedCellId = (--firstGene_NodeIndexToId.end())->second;
    auto& lastConstructedCell = phenotype.getCellRef(lastConstructedCellId);

    std::optional<uint64_t> prevLastConstructedCellId;
    if (firstGene_NodeIndexToId.size() > 1) {
        prevLastConstructedCellId = (--(--firstGene_NodeIndexToId.end()))->second;
    } else {
        if (lastConstructedCell.getCellType() == CellType_Constructor) {
            auto const& constructor = std::get<ConstructorDescription>(lastConstructedCell._cellTypeData);
            auto geneIndex = constructor._geneIndex;
            if (geneAndNodeIndexToId.contains(geneIndex)) {
                auto& secondGene_NodeIndexToId = geneAndNodeIndexToId.at(geneIndex);
                prevLastConstructedCellId = (--secondGene_NodeIndexToId.end())->second;
            }
        }
    }

    if (prevLastConstructedCellId.has_value()) {
        auto& prevLastConstructedCell = phenotype.getCellRef(prevLastConstructedCellId.value());

        auto angle = Math::angleOfVector(prevLastConstructedCell._pos - lastConstructedCell._pos);
        editService.rotate(phenotype, -angle);
    }

    // Create preview cells
    phenotype.forEachCell([&](CellDescription const& cell) {
        int color = cell._color; // Default to cell's own color
        if (cell._geneIndex < genome._genes.size() && cell._nodeIndex < genome._genes.at(cell._geneIndex)._nodes.size()) {
            color = genome._genes.at(cell._geneIndex)._nodes.at(cell._nodeIndex)._color;
        }
        result._cells.push_back(CellPreviewDescription().pos(cell._pos).color(color).geneIndex(cell._geneIndex).nodeIndex(cell._nodeIndex));
    });

    // Create preview connections
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    std::map<std::pair<uint64_t, uint64_t>, bool> arrowFromCell1ToCell2; // key: (cellId1, cellId2), value: has arrow from cell1 to cell2
    
    // First pass: determine arrow directions for each cell
    phenotype.forEachCell([&](CellDescription const& cell) {
        // Calculate signal routing restriction angles (similar to CUDA logic)
        auto signalAngleRestrictionStart = 180.0f + cell._signalRoutingRestriction._baseAngle - cell._signalRoutingRestriction._openingAngle / 2;
        auto signalAngleRestrictionEnd = 180.0f + cell._signalRoutingRestriction._baseAngle + cell._signalRoutingRestriction._openingAngle / 2;
        signalAngleRestrictionStart = Math::normalizedAngle(signalAngleRestrictionStart, 0.0f);
        signalAngleRestrictionEnd = Math::normalizedAngle(signalAngleRestrictionEnd, 0.0f);
        
        // Calculate summed angles and check each connection
        auto summedAngle = 0.0f;
        for (int i = 0; i < cell._connections.size(); ++i) {
            if (i > 0) {
                summedAngle += cell._connections[i]._angleFromPrevious;
            }
            auto connectedCellId = cell._connections[i]._cellId;
            
            // Check if arrow should be drawn (similar to CUDA logic lines 568-569)
            bool shouldDrawArrow = !cell._signalRoutingRestriction._active 
                || Math::isAngleStrictInBetween(signalAngleRestrictionStart, signalAngleRestrictionEnd, summedAngle);
            
            if (shouldDrawArrow) {
                arrowFromCell1ToCell2[{cell._id, connectedCellId}] = true;
            }
        }
    });
    
    // Second pass: create connections with arrow information
    phenotype.forEachCell([&](CellDescription const& cell) {
        for (const auto& connection : cell._connections) {
            uint64_t cellId1 = cell._id;
            uint64_t cellId2 = connection._cellId;

            auto connectionPair = std::make_pair(std::min(cellId1, cellId2), std::max(cellId1, cellId2));
            if (processedConnections.find(connectionPair) != processedConnections.end()) {
                continue;
            }
            processedConnections.insert(connectionPair);

            // Determine arrow directions
            bool arrowToCell1 = arrowFromCell1ToCell2.count({cellId2, cellId1}) > 0; // arrow FROM cell2 TO cell1
            bool arrowToCell2 = arrowFromCell1ToCell2.count({cellId1, cellId2}) > 0; // arrow FROM cell1 TO cell2

            ConnectionPreviewDescription previewConnection;
            previewConnection.cell1(phenotype.getCellRef(cellId1, cache)._pos)
                            .cell2(phenotype.getCellRef(cellId2, cache)._pos)
                            .arrowToCell1(arrowToCell1)
                            .arrowToCell2(arrowToCell2);
            result._connections.push_back(previewConnection);
        }
    });

    return result;
}

PreviewDescription PreviewDescriptionConverterService::convert(CollectionDescription&& phenotype) const
{
    // Create a default empty genome for the single-parameter version
    GenomeDescription emptyGenome;
    return convert(emptyGenome, std::move(phenotype));
}
