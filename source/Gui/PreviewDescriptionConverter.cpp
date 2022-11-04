#include "PreviewDescriptionConverter.h"

#include "Base/Math.h"

PreviewDescription PreviewDescriptionConverter::convert(GenomeDescription const& genome)
{
    PreviewDescription result;

    bool first = true;
    RealVector2D pos(0, 0);
    RealVector2D direction(0, 1);
    for (auto const& cell : genome) {
        RealVector2D prevPos = pos;

        if (!first) {
            pos += Math::rotateClockwise(-direction, cell.referenceAngle) * cell.referenceDistance;
        }
        result.cells.emplace_back(pos, cell.color);
        if (!first) {
            result.connections.emplace_back(prevPos, pos);
        }

        first = false;
        direction = pos - prevPos;
        Math::normalize(direction);
    }

    return result;
}
