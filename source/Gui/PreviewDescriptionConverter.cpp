#include "PreviewDescriptionConverter.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

namespace
{
    float constexpr OffspringCellDistance = 1.6f;

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

        PreviewDescription result;

        RealVector2D pos;
        RealVector2D direction(0, -1);

        std::unordered_map<IntVector2D, std::vector<CellPreviewDescriptionIntern>> cellsInternBySlot;
        std::vector<CellPreviewDescriptionIntern> cellsIntern;
        int index = 0;
        for (auto const& node : genome) {
            RealVector2D prevPos = pos;

            if (index > 0) {
                pos += direction * node.referenceDistance;
            }

            CellPreviewDescription cell{.pos = pos, .color = node.color};
            if (std::holds_alternative<int>(selectedNode)) {
                cell.selected = index == std::get<int>(selectedNode);
            }
            if (std::holds_alternative<SelectAllNodes>(selectedNode)) {
                cell.selected = true;
            }

            result.cells.emplace_back(cell);
            if (index > 0) {
                direction = Math::rotateClockwise(-direction, node.referenceAngle);
                result.connections.emplace_back(prevPos, pos);
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
                    auto const& findResult = cellsInternBySlot.find({intPos.x + dx, intPos.y + dy});
                    if (findResult != cellsInternBySlot.end()) {
                        for (auto& otherCell : findResult->second) {
                            if (Math::length(otherCell.pos - pos) < OffspringCellDistance) {
                                if (otherCell.connectionIndices.size() < parameters.cellMaxBonds
                                    && cellIntern.connectionIndices.size() < parameters.cellMaxBonds) {
                                    result.connections.emplace_back(otherCell.pos, pos);
                                    cellIntern.connectionIndices.insert(otherCell.nodeIndex);
                                    otherCell.connectionIndices.insert(index);
                                }
                            }
                        }
                    }
                }
            }

            cellsInternBySlot[intPos].emplace_back(cellIntern);
            cellsIntern.emplace_back(cellIntern);
            ++index;
        }

        //process sub genomes
        for (auto const& [node, cellIntern] : boost::combine(genome, cellsIntern)) {
            if (node.getCellFunctionType() == Enums::CellFunction_Constructor) {
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
                    auto connectedCellIntern = cellsIntern.at(connectedCellIndex);
                    angles.emplace_back(Math::angleOfVector(connectedCellIntern.pos - cellIntern.pos));
                }

                //sort angles
                if (angles.size() > 2) {
                    int pos = 0;
                    int numAngles = toInt(angles.size());
                    do {
                        auto angle0 = angles.at(pos % numAngles);
                        auto angle1 = angles.at((pos + 1) % numAngles);
                        auto angle2 = angles.at((pos + 2) % numAngles);
                        if (Math::isAngleInBetween(angle0, angle2, angle1)) {
                            ++pos;
                        } else {
                            angles.at((pos + 1) % numAngles) = angle2;
                            angles.at((pos + 2) % numAngles) = angle1;
                            --pos;
                        }
                    } while (pos <= numAngles);
                }

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
                insert(result, previewPart);
                if (!constructor.separateConstruction) {
                    result.connections.emplace_back(previewPart.cells.back().pos, cellIntern.pos);
                }
            }

            ++index;
        }

        //transform to desired position and angle
        if (desiredEndAngle) {
            auto actualEndAngle = Math::angleOfVector(direction);
            auto angleDiff = Math::subtractAngle(*desiredEndAngle, actualEndAngle);
            rotate(result, result.cells.back().pos, angleDiff + 180.0f);
        }
        if (desiredEndPos) {
            translate(result, *desiredEndPos - result.cells.back().pos);
        }
        return result;
    }
}

PreviewDescription
PreviewDescriptionConverter::convert(GenomeDescription const& genome, std::optional<int> selectedNode, SimulationParameters const& parameters)
{
    return convertIntern(genome, selectedNode ? SelectNode(*selectedNode) : SelectNode(SelectNoNodes()), std::nullopt, std::nullopt, parameters);
}
