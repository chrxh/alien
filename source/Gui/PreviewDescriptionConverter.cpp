#include "PreviewDescriptionConverter.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

namespace
{
    void rotate(PreviewDescription& preview, RealVector2D const& center, float angle)
    {
        auto rotMatrix = Math::calcRotationMatrix(angle);

        for (auto& cell : preview.cells) {
            cell.pos = rotMatrix * (cell.pos - center) + center;
        }
        for (auto& connection : preview.connections) {
            connection.cell1 = rotMatrix * (connection.cell1 - center) + center;
            connection.cell2 = rotMatrix * (connection.cell2 - center) + center;
        }
    }

    void translate(PreviewDescription& preview, RealVector2D const& delta)
    {
        for (auto& cell : preview.cells) {
            cell.pos += delta;
        }
        for (auto& connection : preview.connections) {
            connection.cell1 += delta;
            connection.cell2 += delta;
        }
    }

    void insert(PreviewDescription& target, PreviewDescription const& source)
    {
        target.cells.insert(target.cells.begin(), source.cells.begin(), source.cells.end());
        target.connections.insert(target.connections.begin(), source.connections.begin(), source.connections.end());
    }

    struct CellPreviewDescriptionIntern
    {
        int nodeIndex = 0;
        bool selected = false;
        RealVector2D pos;
        std::set<int> connectionIndices;
    };
    struct SelectAllNodes
    {};
    struct SelectNoNodes
    {};
    using SelectNode = std::variant<SelectAllNodes, SelectNoNodes, int>;


    struct ProcessedGenomeDescriptionResult
    {
        PreviewDescription preview;
        std::vector<CellPreviewDescriptionIntern> cellsIntern;
        RealVector2D direction;
    };
    ProcessedGenomeDescriptionResult processGenomeDescription(GenomeDescription const& genome, SelectNode selectedNode, SimulationParameters const& parameters)
    {
        ProcessedGenomeDescriptionResult result;
        result.direction = RealVector2D{0, -1};

        RealVector2D pos;
        std::unordered_map<IntVector2D, std::vector<int>> cellInternIndicesBySlot;
        int index = 0;
        for (auto const& node : genome) {
            RealVector2D prevPos = pos;

            if (index > 0) {
                pos += result.direction * node.referenceDistance;
            }

            CellPreviewDescription cell{.pos = pos, .color = node.color};
            if (std::holds_alternative<int>(selectedNode)) {
                cell.selected = index == std::get<int>(selectedNode);
            }
            if (std::holds_alternative<SelectAllNodes>(selectedNode)) {
                cell.selected = true;
            }

            result.preview.cells.emplace_back(cell);
            if (index > 0) {
                result.direction = Math::rotateClockwise(-result.direction, -(180.0f - node.referenceAngle));
                result.preview.connections.emplace_back(prevPos, pos);
            }

            //create cell description intern
            CellPreviewDescriptionIntern cellIntern;
            cellIntern.nodeIndex = index;
            cellIntern.selected = cell.selected;
            cellIntern.pos = pos;
            if (index > 0) {
                cellIntern.connectionIndices.insert(index - 1);
            }
            if (index < genome.size() - 1) {
                cellIntern.connectionIndices.insert(index + 1);
            }
            IntVector2D intPos{toInt(pos.x), toInt(pos.y)};
            for (int dx = -2; dx <= 2; ++dx) {
                for (int dy = -2; dy <= 2; ++dy) {
                    auto const& findResult = cellInternIndicesBySlot.find({intPos.x + dx, intPos.y + dy});
                    if (findResult != cellInternIndicesBySlot.end()) {
                        for (auto const& otherCellIndex : findResult->second) {
                            auto& otherCell = result.cellsIntern.at(otherCellIndex);
                            if (Math::length(otherCell.pos - pos) < parameters.cellFunctionConstructorConnectingCellMaxDistance) {
                                if (otherCell.connectionIndices.size() < parameters.cellMaxBonds
                                    && cellIntern.connectionIndices.size() < parameters.cellMaxBonds) {
                                    result.preview.connections.emplace_back(otherCell.pos, pos);
                                    cellIntern.connectionIndices.insert(otherCell.nodeIndex);
                                    otherCell.connectionIndices.insert(index);
                                }
                            }
                        }
                    }
                }
            }

            cellInternIndicesBySlot[intPos].emplace_back(toInt(result.cellsIntern.size()));
            result.cellsIntern.emplace_back(cellIntern);
            ++index;
        }
        return std::move(result);
    }

    PreviewDescription convertIntern(
        GenomeDescription const& genome,
        SelectNode selectedNode,
        std::optional<RealVector2D> const& desiredEndPos,
        std::optional<float> const& desiredEndAngle,
        SimulationParameters const& parameters)
    {
        if (genome.empty()) {
            return PreviewDescription();
        }

        ProcessedGenomeDescriptionResult result = processGenomeDescription(genome, selectedNode, parameters);

        //process sub genomes
        for (auto const& [node, cellIntern] : boost::combine(genome, result.cellsIntern)) {
            if (node.getCellFunctionType() == CellFunction_Constructor) {
                auto const& constructor = std::get<ConstructorGenomeDescription>(*node.cellFunction);
                if (constructor.isMakeGenomeCopy()) {
                    continue;
                }
                auto data = constructor.getGenomeData();
                if (data.empty()) {
                    continue;
                }
                auto subGenome = GenomeDescriptionConverter::convertBytesToDescription(data, parameters);

                //angles of connected cells
                std::vector<float> angles;
                for (auto const& connectedCellIndex : cellIntern.connectionIndices) {
                    auto connectedCellIntern = result.cellsIntern.at(connectedCellIndex);
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

                auto direction = Math::unitVectorOfAngle(targetAngle);
                auto previewPart = convertIntern(
                    subGenome,
                    cellIntern.selected ? SelectNode(SelectAllNodes()) : SelectNode(SelectNoNodes()),
                    cellIntern.pos + direction,
                    targetAngle,
                    parameters);
                insert(result.preview, previewPart);
                if (!constructor.separateConstruction) {
                    result.preview.connections.emplace_back(previewPart.cells.back().pos, cellIntern.pos);
                }
            }
        }

        //transform to desired position and angle
        if (desiredEndAngle) {
            auto actualEndAngle = Math::angleOfVector(result.direction);
            auto angleDiff = Math::subtractAngle(*desiredEndAngle, actualEndAngle);
            rotate(result.preview, result.preview.cells.back().pos, angleDiff + 180.0f);
        }
        if (desiredEndPos) {
            translate(result.preview, *desiredEndPos - result.preview.cells.back().pos);
        }
        return result.preview;
    }
}

PreviewDescription
PreviewDescriptionConverter::convert(GenomeDescription const& genome, std::optional<int> selectedNode, SimulationParameters const& parameters)
{
    return convertIntern(genome, selectedNode ? SelectNode(*selectedNode) : SelectNode(SelectNoNodes()), std::nullopt, std::nullopt, parameters);
}
