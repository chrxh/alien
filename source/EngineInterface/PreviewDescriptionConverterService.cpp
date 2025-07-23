#include "PreviewDescriptionConverterService.h"

#include <set>
#include <unordered_map>

#include "DescriptionEditService.h"

PreviewDescription PreviewDescriptionConverterService::convert(CollectionDescription const& data) const
{
    PreviewDescription result;
    
    // Create a temporary collection to use for centering
    CollectionDescription tempData = data;
    
    // Create a cache for efficient cell lookups
    auto cache = tempData.createCache();
    
    // Collect all cells from both direct cells and creature cells
    std::vector<CellDescription> allCells;
    
    // Add direct cells
    for (const auto& cell : tempData._cells) {
        allCells.push_back(cell);
    }
    
    // Add cells from creatures
    for (const auto& creature : tempData._creatures) {
        for (const auto& cell : creature._cells) {
            allCells.push_back(cell);
        }
    }
    
    // If no cells, return empty result
    if (allCells.empty()) {
        return result;
    }
    
    // Center the temporary data at {0, 0}
    DescriptionEditService::get().setCenter(tempData, {0.0f, 0.0f});
    
    // Create a mapping from original cell IDs to the centered positions
    std::unordered_map<uint64_t, RealVector2D> centeredPositions;
    
    // First collect all cell positions after centering
    for (const auto& cell : tempData._cells) {
        centeredPositions[cell._id] = cell._pos;
    }
    
    for (const auto& creature : tempData._creatures) {
        for (const auto& cell : creature._cells) {
            centeredPositions[cell._id] = cell._pos;
        }
    }
    
    // Convert cells to CellPreviewDescription
    for (const auto& cell : allCells) {
        CellPreviewDescription previewCell;
        previewCell.pos = centeredPositions[cell._id];
        previewCell.color = cell._color;
        previewCell.nodeIndex = cell._genomeNodeIndex;
        result.cells.push_back(previewCell);
    }
    
    // Convert connections to ConnectionPreviewDescription
    std::set<std::pair<uint64_t, uint64_t>> processedConnections;
    
    for (const auto& cell : allCells) {
        for (const auto& connection : cell._connections) {
            uint64_t cellId1 = cell._id;
            uint64_t cellId2 = connection._cellId;
            
            // Avoid duplicate connections (since connections are bidirectional)
            auto connectionPair = std::make_pair(std::min(cellId1, cellId2), std::max(cellId1, cellId2));
            if (processedConnections.find(connectionPair) != processedConnections.end()) {
                continue;
            }
            processedConnections.insert(connectionPair);
            
            // Find the connected cell's position
            auto pos1It = centeredPositions.find(cellId1);
            auto pos2It = centeredPositions.find(cellId2);
            
            if (pos1It != centeredPositions.end() && pos2It != centeredPositions.end()) {
                ConnectionPreviewDescription previewConnection;
                previewConnection.cell1 = pos1It->second;
                previewConnection.cell2 = pos2It->second;
                previewConnection.arrowToCell1 = false;  // No arrow direction specified in requirements
                previewConnection.arrowToCell2 = false;
                result.connections.push_back(previewConnection);
            }
        }
    }
    
    return result;
}
