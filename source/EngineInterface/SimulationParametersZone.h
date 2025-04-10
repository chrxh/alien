#pragma once

#include <cstdint>

#include "SimulationParametersTypes.h"

using Orientation = int;
enum Orientation_
{
    Orientation_Clockwise,
    Orientation_CounterClockwise
};

using ForceField = int;
enum ForceField_
{
    ForceField_None,
    ForceField_Radial,
    ForceField_Central,
    ForceField_Linear
};

using ZoneShapeType = int;
enum ZoneShapeType_
{
    ZoneShapeType_Circular,
    ZoneShapeType_Rectangular
};

struct SimulationParametersZone
{
    int locationIndex = -1;

    bool operator==(SimulationParametersZone const&) const = default;
};
