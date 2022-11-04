#include "PreviewDescriptionConverter.h"

#include "Base/Math.h"

namespace
{
    float constexpr OffspringCellDistance = 1.6f;

    struct CellPreviewDescriptionIntern
    {
        RealVector2D pos;
        int numConnections = 0;
    };
}

PreviewDescription PreviewDescriptionConverter::convert(GenomeDescription const& genome, SimulationParameters const& parameters)
{
    PreviewDescription result;

    RealVector2D pos(0, 0);
    RealVector2D direction(0, 1);

    std::unordered_map<IntVector2D, std::vector<CellPreviewDescriptionIntern>> cellsBySlot;
    int index = 0;
    for (auto const& cell : genome) {
        RealVector2D prevPos = pos;

        if (index > 0) {
            pos += direction * cell.referenceDistance;
        }
        result.cells.emplace_back(pos, cell.color);
        if (index > 0) {
            direction = Math::rotateClockwise(-direction, cell.referenceAngle);
            result.connections.emplace_back(prevPos, pos);
        }

        IntVector2D intPos{toInt(pos.x), toInt(pos.y)};
        for (int dx = - 2; dx <= 2; ++dx) {
            for (int dy = -2; dy <= 2; ++dy) {
                auto const& findResult = cellsBySlot.find({intPos.x + dx, intPos.y + dy});
                if (findResult != cellsBySlot.end()) {
                    for (auto & otherCell : findResult->second) {
                        if (Math::length(otherCell.pos - pos) < OffspringCellDistance) {
                            if (otherCell.numConnections < parameters.cellMaxBonds) {
                                result.connections.emplace_back(otherCell.pos, pos);
                                ++otherCell.numConnections;
                            }
                        }
                    }
                }
            }
        }

        int numConnections = index == 0 || index == genome.size() ? 1 : 2;
        cellsBySlot[intPos].emplace_back(CellPreviewDescriptionIntern{pos, numConnections});

        ++index;
    }

    return result;
}
