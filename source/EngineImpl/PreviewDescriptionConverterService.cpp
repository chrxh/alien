#include "PreviewDescriptionConverterService.h"

#include <set>

#include "EngineInterface/DescriptionEditService.h"

PreviewDescription PreviewDescriptionConverterService::convert(CollectionDescription&& data) const
{
    PreviewDescription result;
    
    auto cache = data.createCache();
    auto const& editService = DescriptionEditService::get();

    // Remove seed
    uint64_t smallestCellId = 0xffffffffffffffff;
    data.forEachCell([&smallestCellId](auto const& cell) { smallestCellId = std::min(smallestCellId, cell._id); });
    editService.removeCell(data, smallestCellId);
    if (data.isEmpty()) {
        return result;
    }

    // Get last constructed cell on principal gene
    std::map<int, std::map<int, uint64_t>> geneAndNodeIndexToId;
    data.forEachCell([&geneAndNodeIndexToId](auto const& cell) { geneAndNodeIndexToId[cell._geneIndex][cell._nodeIndex] = cell._id; });

    auto lastConstructedCellId = (--geneAndNodeIndexToId.at(0).end())->second;
    auto& lastConstructedCell = data.getCellRef(lastConstructedCellId);

    data.forEachCell([&](CellDescription const& cell) {
        result._cells.push_back(CellPreviewDescription().pos(cell._pos).color(cell._color).geneIndex(cell._geneIndex).nodeIndex(cell._nodeIndex));
    });
    
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    
    data.forEachCell([&](CellDescription const& cell) {
        for (const auto& connection : cell._connections) {
            uint64_t cellId1 = cell._id;
            uint64_t cellId2 = connection._cellId;
            
            auto connectionPair = std::make_pair(std::min(cellId1, cellId2), std::max(cellId1, cellId2));
            if (processedConnections.find(connectionPair) != processedConnections.end()) {
                continue;
            }
            processedConnections.insert(connectionPair);
            
            ConnectionPreviewDescription previewConnection;
            previewConnection.cell1(data.getCellRef(cellId1, cache)._pos).cell2(data.getCellRef(cellId2, cache)._pos).arrowToCell1(false).arrowToCell2(false);
            result._connections.push_back(previewConnection);
        }
    });
    
    return result;
}
