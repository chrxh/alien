#include "PreviewDescriptionConverterService.h"

#include <set>

#include "Base/Math.h"

#include "EngineInterface/DescriptionEditService.h"

PreviewDescription PreviewDescriptionConverterService::convert(CollectionDescription&& data) const
{
    PreviewDescription result;

    auto const& editService = DescriptionEditService::get();

    // Remove seed
    uint64_t smallestCellId = 0xffffffffffffffff;
    data.forEachCell([&smallestCellId](auto const& cell) { smallestCellId = std::min(smallestCellId, cell._id); });
    editService.removeCell(data, smallestCellId);
    auto cache = data.createCache();
    if (data.isEmpty()) {
        return result;
    }

    // Center
    editService.setCenter(data, {0.0f, 0.0f});

    // Try to get last and previous last constructed cell on principal gene
    std::map<int, std::map<int, uint64_t>> geneAndNodeIndexToId;
    data.forEachCell([&geneAndNodeIndexToId](auto const& cell) { geneAndNodeIndexToId[cell._geneIndex][cell._nodeIndex] = cell._id; });

    auto& firstGene_NodeIndexToId = geneAndNodeIndexToId.at(0);
    auto lastConstructedCellId = (--firstGene_NodeIndexToId.end())->second;
    auto& lastConstructedCell = data.getCellRef(lastConstructedCellId);

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
        auto& prevLastConstructedCell = data.getCellRef(prevLastConstructedCellId.value());

        auto angle = Math::angleOfVector(prevLastConstructedCell._pos - lastConstructedCell._pos);
        editService.rotate(data, -angle);
    }

    // Create preview cells
    data.forEachCell([&](CellDescription const& cell) {
        result._cells.push_back(CellPreviewDescription().pos(cell._pos).color(cell._color).geneIndex(cell._geneIndex).nodeIndex(cell._nodeIndex));
    });

    // Create preview connections
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
