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

void SpaceCalculator::correctPosition(RealVector2D& pos) const
{
    pos.x = std::fmod(std::fmod(pos.x, _worldSizeFloat.x) + _worldSizeFloat.x, _worldSizeFloat.x);
    pos.y = std::fmod(std::fmod(pos.y, _worldSizeFloat.y) + _worldSizeFloat.y, _worldSizeFloat.y);
}
