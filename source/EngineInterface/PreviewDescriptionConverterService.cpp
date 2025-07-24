#include "PreviewDescriptionConverterService.h"

#include <set>
#include <unordered_map>

#include "DescriptionEditService.h"

PreviewDescription PreviewDescriptionConverterService::convert(CollectionDescription const& data) const
{
    PreviewDescription result;
    
    CollectionDescription tempData = data;
    auto cache = tempData.createCache();
    
    bool hasCells = false;
    tempData.forEach([&](CellDescription const& cell) {
        hasCells = true;
    });
    
    if (!hasCells) {
        return result;
    }
    
    DescriptionEditService::get().setCenter(tempData, {0.0f, 0.0f});
    
    std::unordered_map<uint64_t, RealVector2D> centeredPositions;
    tempData.forEach([&](CellDescription const& cell) {
        centeredPositions[cell._id] = cell._pos;
        
        CellPreviewDescription previewCell;
        previewCell.pos(cell._pos).color(cell._color).nodeIndex(cell._genomeNodeIndex);
        result._cells.push_back(previewCell);
    });
    
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    
    tempData.forEach([&](CellDescription const& cell) {
        for (const auto& connection : cell._connections) {
            uint64_t cellId1 = cell._id;
            uint64_t cellId2 = connection._cellId;
            
            auto connectionPair = std::make_pair(std::min(cellId1, cellId2), std::max(cellId1, cellId2));
            if (processedConnections.find(connectionPair) != processedConnections.end()) {
                continue;
            }
            processedConnections.insert(connectionPair);
            
            auto pos1It = centeredPositions.find(cellId1);
            auto pos2It = centeredPositions.find(cellId2);
            
            if (pos1It != centeredPositions.end() && pos2It != centeredPositions.end()) {
                ConnectionPreviewDescription previewConnection;
                previewConnection.cell1(pos1It->second).cell2(pos2It->second).arrowToCell1(false).arrowToCell2(false);
                result._connections.push_back(previewConnection);
            }
        }
    });
    
    return result;
}
