#include "PreviewDescriptionConverterService.h"

#include <set>

#include "DescriptionEditService.h"

PreviewDescription PreviewDescriptionConverterService::convert(CollectionDescription data) const
{
    PreviewDescription result;
    
    auto cache = data.createCache();

    auto const& editService = DescriptionEditService::get();

    uint64_t smallestCellId = 0xffffffffffffffff;
    data.forEach([&](auto const& cell) { smallestCellId = std::min(smallestCellId, cell._id); });
    editService.removeCell(data, smallestCellId);
    editService.setCenter(data, {0.0f, 0.0f});
    
    data.forEach([&](CellDescription const& cell) {
        CellPreviewDescription previewCell;
        previewCell.pos(cell._pos).color(cell._color).nodeIndex(cell._genomeNodeIndex);
        result._cells.push_back(previewCell);
    });
    
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    
    data.forEach([&](CellDescription const& cell) {
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
