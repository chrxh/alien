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

PreviewDescription PreviewDescriptionConverterService::convert(GenomeDescription const& genome, CollectionDescription&& phenotype, int startGeneIndex) const
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
        editService.rotate(phenotype, -angle);
    }

    // Create preview cells
    phenotype.forEachCell([&](CellDescription const& cell) {
        auto const& color = genome._genes.at(cell._geneIndex)._nodes.at(cell._nodeIndex)._color;
        result._cells.push_back(CellPreviewDescription().pos(cell._pos).color(color).geneIndex(cell._geneIndex).nodeIndex(cell._nodeIndex));
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

            ConnectionPreviewDescription previewConnection;
            previewConnection.cell1(phenotype.getCellRef(cellId1, cache)._pos).cell2(phenotype.getCellRef(cellId2, cache)._pos).arrowToCell1(false).arrowToCell2(false);
            result._connections.push_back(previewConnection);
        }
    });

    return result;
}
