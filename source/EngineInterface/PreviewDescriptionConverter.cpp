#include "PreviewDescriptionConverter.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

namespace
{
    struct CellPreviewDescriptionIntern
    {
        RealVector2D pos;
        int nodeIndex = 0;
        int executionOrderNumber = 0;
        bool inputBlocked = false;
        int inputExecutionOrderNumber = 0;
        bool outputBlocked = false;
        int color = 0;
        std::set<int> connectionIndices;
    };

    void insert(std::vector<CellPreviewDescriptionIntern>& target, std::vector<CellPreviewDescriptionIntern> const& source)
    {
        auto offset = toInt(source.size());
        for (auto& cell : target) {
            std::set<int> newConnectionIndices;
            for (auto const& connectionIndex : cell.connectionIndices) {
                newConnectionIndices.insert(connectionIndex + offset);
            }
            cell.connectionIndices = newConnectionIndices;
        }
        target.insert(target.begin(), source.begin(), source.end());
    }

    void rotate(std::vector<CellPreviewDescriptionIntern>& cells, RealVector2D const& center, float angle)
    {
        auto rotMatrix = Math::calcRotationMatrix(angle);

        for (auto& cell : cells) {
            cell.pos = rotMatrix * (cell.pos - center) + center;
        }
    }

    void translate(std::vector<CellPreviewDescriptionIntern>& cells, RealVector2D const& delta)
    {
        for (auto& cell : cells) {
            cell.pos += delta;
        }
    }

    bool noOverlappingConnection(
        std::vector<CellPreviewDescriptionIntern> const& cells,
        CellPreviewDescriptionIntern const& cell1,
        CellPreviewDescriptionIntern const& cell2)
    {
        auto n = cell1.connectionIndices.size();
        if (n < 2) {
            return true;
        }
        for (auto const& connectionIndex : cell1.connectionIndices) {
            if (connectionIndex >= cells.size()) {
                continue;
            }
            auto connectedCell = cells.at(connectionIndex);
            std::set<int> otherConnectionIndices;
            std::set_intersection(
                cell1.connectionIndices.begin(),
                cell1.connectionIndices.end(),
                connectedCell.connectionIndices.begin(),
                connectedCell.connectionIndices.end(),
                std::inserter(otherConnectionIndices, otherConnectionIndices.begin()));

            for (auto const& otherConnectionIndex : otherConnectionIndices) {
                if (otherConnectionIndex >= cells.size()) {
                    continue;
                }
                auto otherConnectedCell = cells.at(otherConnectionIndex);
                if (Math::crossing(cell1.pos, cell2.pos, connectedCell.pos, otherConnectedCell.pos)) {
                    return false;
                }
            }
        }
        return true;
    }

    struct ProcessedGenomeDescriptionResult
    {
        std::vector<CellPreviewDescriptionIntern> cellsIntern;
        RealVector2D direction;
    };
    ProcessedGenomeDescriptionResult processMainGenomeDescription(GenomeDescription const& genome, std::optional<int> nodeIndex, SimulationParameters const& parameters)
    {
        ProcessedGenomeDescriptionResult result;
        result.direction = RealVector2D{0, 1};

        RealVector2D pos;
        std::unordered_map<IntVector2D, std::vector<int>> cellInternIndicesBySlot;
        int index = 0;
        for (auto const& node : genome) {
            RealVector2D prevPos = pos;

            if (index > 0) {
                pos += result.direction;
            }

            if (index > 0) {
                result.direction = Math::rotateClockwise(-result.direction, (180.0f + node.referenceAngle));
            }

            //create cell description intern
            CellPreviewDescriptionIntern cellIntern;
            cellIntern.color = node.color;
            cellIntern.inputBlocked = node.inputBlocked;
            cellIntern.inputExecutionOrderNumber = node.inputExecutionOrderNumber;
            cellIntern.outputBlocked = node.outputBlocked;
            cellIntern.executionOrderNumber = node.executionOrderNumber;
            cellIntern.nodeIndex = nodeIndex ? *nodeIndex : index;
            cellIntern.pos = pos;
            if (index > 0) {
                cellIntern.connectionIndices.insert(index - 1);
            }
            if (index < genome.size() - 1) {
                cellIntern.connectionIndices.insert(index + 1);
            }

            //find nearby cells
            std::vector<int> nearbyCellIndices;
            IntVector2D intPos{toInt(pos.x), toInt(pos.y)};
            for (int dx = -2; dx <= 2; ++dx) {
                for (int dy = -2; dy <= 2; ++dy) {
                    auto const& findResult = cellInternIndicesBySlot.find({intPos.x + dx, intPos.y + dy});
                    if (findResult != cellInternIndicesBySlot.end()) {
                        for (auto const& otherCellIndex : findResult->second) {
                            auto& otherCell = result.cellsIntern.at(otherCellIndex);
                            if (otherCellIndex != index && Math::length(otherCell.pos - pos) < parameters.cellFunctionConstructorConnectingCellMaxDistance) {
                                if (otherCell.connectionIndices.size() < parameters.cellMaxBonds
                                    && cellIntern.connectionIndices.size() < parameters.cellMaxBonds) {
                                    nearbyCellIndices.emplace_back(otherCellIndex);
                                }
                            }
                        }
                    }
                }
            }

            //sort by distance
            std::sort(nearbyCellIndices.begin(), nearbyCellIndices.end(), [&](int index1, int index2) {
                auto const& otherCell1 = result.cellsIntern.at(index1);
                auto const& otherCell2 = result.cellsIntern.at(index2);
                return Math::length(otherCell1.pos - pos) < Math::length(otherCell2.pos - pos);
            });

            //add connections
            for (auto const& otherCellIndex : nearbyCellIndices) {
                auto& otherCell = result.cellsIntern.at(otherCellIndex);
                if (noOverlappingConnection(result.cellsIntern, cellIntern, otherCell) && noOverlappingConnection(result.cellsIntern, otherCell, cellIntern)) {
                    cellIntern.connectionIndices.insert(otherCellIndex);
                    otherCell.connectionIndices.insert(index);
                }
            }

            cellInternIndicesBySlot[intPos].emplace_back(toInt(result.cellsIntern.size()));
            result.cellsIntern.emplace_back(cellIntern);
            ++index;
        }
        return result;
    }

    std::vector<CellPreviewDescriptionIntern> processGenomeDescription(
        GenomeDescription const& genome,
        std::optional<int> nodeIndex,
        std::optional<RealVector2D> const& desiredEndPos,
        std::optional<float> const& desiredEndAngle,
        SimulationParameters const& parameters)
    {
        if (genome.empty()) {
            return {};
        }

        ProcessedGenomeDescriptionResult processedGenome = processMainGenomeDescription(genome, nodeIndex, parameters);

        std::vector<CellPreviewDescriptionIntern> result = processedGenome.cellsIntern;

        //process sub genomes
        size_t indexOffset = 0;
        int index = 0;
        for (auto const& [node, cellIntern] : boost::combine(genome, processedGenome.cellsIntern)) {
            if (node.getCellFunctionType() == CellFunction_Constructor) {
                auto const& constructor = std::get<ConstructorGenomeDescription>(*node.cellFunction);
                if (constructor.isMakeGenomeCopy()) {
                    ++index;
                    continue;
                }
                auto data = constructor.getGenomeData();
                if (data.empty()) {
                    ++index;
                    continue;
                }
                auto subGenome = GenomeDescriptionConverter::convertBytesToDescription(data);

                //angles of connected cells
                std::vector<float> angles;
                for (auto const& connectedCellIndex : cellIntern.connectionIndices) {
                    auto connectedCellIntern = processedGenome.cellsIntern.at(connectedCellIndex);
                    angles.emplace_back(Math::angleOfVector(connectedCellIntern.pos - cellIntern.pos));
                }
                std::ranges::sort(angles);

                //find largest diff
                float targetAngle = 0;
                if (angles.size() > 1) {
                    std::optional<float> largestAngleDiff;
                    int pos = 0;
                    int numAngles = toInt(angles.size());
                    do {
                        auto angle0 = angles.at(pos % numAngles);
                        auto angle1 = angles.at((pos + 1) % numAngles);
                        auto angleDiff = Math::subtractAngle(angle1, angle0);
                        if (!largestAngleDiff.has_value() || (angleDiff > *largestAngleDiff)) {
                            largestAngleDiff = angleDiff;
                            targetAngle = angle0 + angleDiff / 2;
                        }
                        ++pos;
                    } while (pos <= numAngles);
                }
                if (angles.size() == 1) {
                    targetAngle = angles.front() + 180.0f;
                }
                targetAngle += subGenome.front().referenceAngle;
                auto direction = Math::unitVectorOfAngle(targetAngle);
                auto previewPart = processGenomeDescription(subGenome, cellIntern.nodeIndex, cellIntern.pos + direction, targetAngle, parameters);
                insert(result, previewPart);
                indexOffset += previewPart.size();
                if (!constructor.separateConstruction) {
                    auto cellIndex1 = previewPart.size() - 1;
                    auto cellIndex2 = index + indexOffset;
                    result.at(cellIndex1).connectionIndices.insert(toInt(cellIndex2));
                    result.at(cellIndex2).connectionIndices.insert(toInt(cellIndex1));
                }
            }
            ++index;
        }

        //transform to desired position and angle
        if (desiredEndAngle) {
            auto actualEndAngle = Math::angleOfVector(processedGenome.direction);
            auto angleDiff = Math::subtractAngle(*desiredEndAngle, actualEndAngle);
            rotate(result, result.back().pos, angleDiff + 180.0f);
        }
        if (desiredEndPos) {
            translate(result, *desiredEndPos - result.back().pos);
        }
        return result;
    }

    int calcInputExecutionOrder(
        std::vector<CellPreviewDescriptionIntern> const& cells,
        CellPreviewDescriptionIntern const& cell,
        SimulationParameters const& parameters)
    {
        if (cell.inputBlocked) {
            return -1;
        }
        return cell.inputExecutionOrderNumber;
    }

    PreviewDescription createPreviewDescription(std::vector<CellPreviewDescriptionIntern> const& cells, SimulationParameters const& parameters)
    {
        PreviewDescription result;
        std::map<std::pair<int, int>, int> cellIndicesToCreatedConnectionIndex;
        int index = 0;
        for (auto const& cell : cells) {
            CellPreviewDescription cellPreview{.pos = cell.pos, .executionOrderNumber = cell.executionOrderNumber, .color = cell.color, .nodeIndex = cell.nodeIndex};
            result.cells.emplace_back(cellPreview);
            auto inputExecutionOrder = calcInputExecutionOrder(cells, cell, parameters);
            for (auto const& connectionIndex : cell.connectionIndices) {
                auto const& otherCell = cells.at(connectionIndex);
                auto findResult = cellIndicesToCreatedConnectionIndex.find(std::pair(index, connectionIndex));
                if (findResult == cellIndicesToCreatedConnectionIndex.end()) {
                    ConnectionPreviewDescription connection;
                    connection.cell1 = cell.pos;
                    connection.cell2 = otherCell.pos;
                    connection.arrowToCell1 = inputExecutionOrder == otherCell.executionOrderNumber && !otherCell.outputBlocked;
                    result.connections.emplace_back(connection);
                    cellIndicesToCreatedConnectionIndex.emplace(std::pair(connectionIndex, index), toInt(result.connections.size() - 1));
                } else {
                    auto connectionIndex = findResult->second;
                    result.connections.at(connectionIndex).arrowToCell2 = inputExecutionOrder == otherCell.executionOrderNumber && !otherCell.outputBlocked;
                }
            }
            ++index;
        }
        return result;
    }
}

PreviewDescription
PreviewDescriptionConverter::convert(GenomeDescription const& genome, std::optional<int> selectedNode, SimulationParameters const& parameters)
{
    auto cellInterDescriptions =
        processGenomeDescription(genome, std::nullopt, std::nullopt, std::nullopt, parameters);
    return createPreviewDescription(cellInterDescriptions, parameters);
}
