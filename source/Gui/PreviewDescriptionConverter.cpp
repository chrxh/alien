#include "PreviewDescriptionConverter.h"

#include <boost/range/combine.hpp>

#include "Base/Math.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

namespace
{
    float constexpr OffspringCellDistance = 1.6f;

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
    void convertIntern(
        PreviewDescription& result,
        GenomeDescription const& genome,
        SelectNode selectedNode,
        std::optional<RealVector2D> const& startPos,
        SimulationParameters const& parameters)
    {
        RealVector2D pos = startPos.value_or(RealVector2D(0, 0));
        RealVector2D direction(0, 1);

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
                auto data = std::get<ConstructorGenomeDescription>(*node.cellFunction).getGenomeData();
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

                //3. find largest diff
                std::optional<float> largestAngleDiff;
                float targetAngle = 0;
                if (angles.size() > 1) {
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

                convertIntern(
                    result,
                    subGenome,
                    cellIntern.selected ? SelectNode(SelectAllNodes()) : SelectNode(SelectNoNodes()),
                    cellIntern.pos + Math::unitVectorOfAngle(targetAngle),
                    parameters);
            }

            ++index;
        }
    }
}

PreviewDescription
PreviewDescriptionConverter::convert(GenomeDescription const& genome, std::optional<int> selectedNode, SimulationParameters const& parameters)
{
    PreviewDescription result;
    convertIntern(result, genome, selectedNode ? SelectNode(*selectedNode) : SelectNode(SelectNoNodes()), std::nullopt, parameters);
    return result;
}
