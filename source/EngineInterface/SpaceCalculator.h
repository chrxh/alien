#pragma once

#include "Base/Definitions.h"
#include "Definitions.h"
#include "DllExport.h"

class SpaceCalculator
{
public:
    ENGINEINTERFACE_EXPORT SpaceCalculator(IntVector2D const& worldSize);
    ENGINEINTERFACE_EXPORT float distance(RealVector2D const& a, RealVector2D const& b) const;

private:
    void correctDisplacement(RealVector2D& displacement) const;
    void correctPosition(RealVector2D& pos) const;

    IntVector2D _worldSize;
    RealVector2D _worldSizeFloat;
};
