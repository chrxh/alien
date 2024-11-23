#include "SpaceCalculator.h"

#include <cmath>

#include "Base/Math.h"

SpaceCalculator::SpaceCalculator(IntVector2D const& worldSize)
    : _worldSize(worldSize)
    , _worldSizeFloat{toFloat(worldSize.x), toFloat(worldSize.y)}
{}

float SpaceCalculator::distance(RealVector2D const& a, RealVector2D const& b) const
{
    auto d = b - a;
    correctDisplacement(d);
    return toFloat(Math::length(d));
}

void SpaceCalculator::correctDisplacement(RealVector2D& displacement) const
{
    RealVector2D intDisplacement{std::floor(displacement.x), std::floor(displacement.y)};
    RealVector2D remainder{displacement.x - intDisplacement.x, displacement.y - intDisplacement.y};
    intDisplacement.x += _worldSizeFloat.x / 2;
    intDisplacement.y += _worldSizeFloat.y / 2;
    correctPosition(intDisplacement);
    intDisplacement.x -= _worldSizeFloat.x / 2;
    intDisplacement.y -= _worldSizeFloat.y / 2;
    displacement.x = intDisplacement.x + remainder.x;
    displacement.y = intDisplacement.y + remainder.y;
}

namespace
{
    void correctIntPosition(IntVector2D& pos, IntVector2D const& worldSize)
    {
        pos = {((pos.x % worldSize.x) + worldSize.x) % worldSize.x, ((pos.y % worldSize.y) + worldSize.y) % worldSize.y};
    }

}

void SpaceCalculator::correctPosition(RealVector2D& pos) const
{
    auto intPart = toIntVector2D(pos);
    auto fracPart = RealVector2D{pos.x - toFloat(intPart.x), pos.y - toFloat(intPart.y)};
    correctIntPosition(intPart, _worldSize);
    pos = {static_cast<float>(intPart.x) + fracPart.x, static_cast<float>(intPart.y) + fracPart.y};
}

RealVector2D SpaceCalculator::getCorrectedPosition(RealVector2D const& pos) const
{
    auto result = pos;
    correctPosition(result);
    return result;
}

RealVector2D SpaceCalculator::getCorrectedDirection(RealVector2D const& pos) const
{
    auto result = pos + RealVector2D{_worldSizeFloat.x / 2, _worldSizeFloat.y / 2};
    correctPosition(result);
    return result - RealVector2D{_worldSizeFloat.x / 2, _worldSizeFloat.y / 2};
}
