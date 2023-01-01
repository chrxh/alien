#pragma once

#include <cstdint>

#include "SimulationParametersSpotValues.h"

using ShapeType = int;
enum SpotShape_
{
    ShapeType_Circular,
    ShapeType_Rectangular
};

using FlowType = int;
enum FlowType_
{
    FlowType_None,
    FlowType_Radial
};

using Orientation = int;
enum Orientation_
{
    Orientation_Clockwise,
    Orientation_CounterClockwise
};

struct RadialFlow
{
    Orientation orientation = Orientation_Clockwise;
    float strength = 0.001f;

    bool operator==(RadialFlow const& other) const { return orientation == other.orientation && strength == other.strength; }
    bool operator!=(RadialFlow const& other) const { return !operator==(other); }
};

union FlowData
{
    RadialFlow radialFlow;
};


struct CircularSpot
{
    float coreRadius = 100.0f;

    bool operator==(CircularSpot const& other) const { return coreRadius == other.coreRadius; }
    bool operator!=(CircularSpot const& other) const { return !operator==(other); }
};

struct RectangularSpot
{
    float width = 100.0f;
    float height = 200.0f;

    bool operator==(RectangularSpot const& other) const { return width == other.width && height == other.height; }
    bool operator!=(RectangularSpot const& other) const { return !operator==(other); }
};

union ShapeData
{
    CircularSpot circularSpot;
    RectangularSpot rectangularSpot;
};

struct SimulationParametersSpot
{
    uint32_t color = 0;
    float posX = 0;
    float posY = 0;

    float fadeoutRadius = 100.0f;

    ShapeType shapeType = ShapeType_Circular;
    ShapeData shapeData = {CircularSpot()};

    FlowType flowType = FlowType_None;
    FlowData flowData = {RadialFlow()};

    SimulationParametersSpotValues values;

    bool operator==(SimulationParametersSpot const& other) const
    {
        if (flowType != other.flowType) {
            return false;
        }
        if (flowType == FlowType_Radial) {
            if (flowData.radialFlow != other.flowData.radialFlow) {
                return false;
            }
        }
        if (shapeType != other.shapeType) {
            return false;
        }
        if (shapeType == ShapeType_Circular) {
            if (shapeData.circularSpot != other.shapeData.circularSpot) {
                return false;
            }
        }
        if (shapeType == ShapeType_Rectangular) {
            if (shapeData.rectangularSpot != other.shapeData.rectangularSpot) {
                return false;
            }
        }

        return color == other.color && posX == other.posX && posY == other.posY && fadeoutRadius == other.fadeoutRadius && values == other.values && shapeType == other.shapeType;
    }
    bool operator!=(SimulationParametersSpot const& other) const { return !operator==(other); }
};
