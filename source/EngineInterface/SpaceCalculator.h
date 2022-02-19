#pragma once

#include "Base/Definitions.h"
#include "Definitions.h"

class SpaceCalculator
{
public:
    SpaceCalculator(IntVector2D const& worldSize);
    float distance(RealVector2D const& a, RealVector2D const& b) const;

    RealVector2D getCorrectedPosition(RealVector2D const& pos) const;

private:
    void correctDisplacement(RealVector2D& displacement) const;
    void correctPosition(RealVector2D& pos) const;

    IntVector2D _worldSize;
    RealVector2D _worldSizeFloat;
};
