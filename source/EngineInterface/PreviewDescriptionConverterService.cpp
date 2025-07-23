#include "PreviewDescriptionConverterService.h"

#include <set>
#include <unordered_map>

#include "DescriptionEditService.h"

PreviewDescription PreviewDescriptionConverterService::convert(CollectionDescription const& data) const
{
    PreviewDescription result;
    
    CollectionDescription tempData = data;
    auto cache = tempData.createCache();
    
    std::vector<CellDescription> allCells;
    
    for (const auto& cell : tempData._cells) {
        allCells.push_back(cell);
    }
    
    for (const auto& creature : tempData._creatures) {
        for (const auto& cell : creature._cells) {
            allCells.push_back(cell);
        }
    }
    
    if (allCells.empty()) {
        return result;
    }
    
    DescriptionEditService::get().setCenter(tempData, {0.0f, 0.0f});
    
    std::unordered_map<uint64_t, RealVector2D> centeredPositions;
    
    for (const auto& cell : tempData._cells) {
        centeredPositions[cell._id] = cell._pos;
    }
    
    for (const auto& creature : tempData._creatures) {
        for (const auto& cell : creature._cells) {
            centeredPositions[cell._id] = cell._pos;
        }
    }
    
    for (const auto& cell : allCells) {
        CellPreviewDescription previewCell;
        previewCell.pos(centeredPositions[cell._id]).color(cell._color).nodeIndex(cell._genomeNodeIndex);
        result._cells.push_back(previewCell);
    }
    
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    
    for (const auto& cell : allCells) {
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
    }
    
    return result;
}
