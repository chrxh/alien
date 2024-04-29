#pragma once

#include <cstdint>
#include <cstring>
#include "SimulationParametersSpotActivatedValues.h"
#include "SimulationParametersSpotValues.h"

using SpotShapeType = int;
enum SpotShapeType_
{
    SpotShapeType_Circular,
    SpotShapeType_Rectangular
};

using FlowType = int;
enum FlowType_
{
    FlowType_None,
    FlowType_Radial,
    FlowType_Central,
    FlowType_Linear
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
    float driftAngle = 0.0f;

    bool operator==(RadialFlow const& other) const { return orientation == other.orientation && strength == other.strength && driftAngle == other.driftAngle; }
    bool operator!=(RadialFlow const& other) const { return !operator==(other); }
};

struct CentralFlow
{
    float strength = 0.05f;

    bool operator==(CentralFlow const& other) const { return strength == other.strength; }
    bool operator!=(CentralFlow const& other) const { return !operator==(other); }
};

struct LinearFlow
{
    float angle = 0;
    float strength = 0.01f;

    bool operator==(LinearFlow const& other) const { return angle == other.angle && strength == other.strength; }
    bool operator!=(LinearFlow const& other) const { return !operator==(other); }
};

union FlowData
{
    RadialFlow radialFlow;
    CentralFlow centralFlow;
    LinearFlow linearFlow;
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

union SpotShapeData
{
    CircularSpot circularSpot;
    RectangularSpot rectangularSpot;
};

static constexpr size_t SIM_PARAM_SPOT_NAME_LENGTH = 32;

struct SimulationParametersSpot
{
    char name[SIM_PARAM_SPOT_NAME_LENGTH]{};
    uint32_t color = 0;
    float posX = 0;
    float posY = 0;
    float velX= 0;
    float velY = 0;

    float fadeoutRadius = 100.0f;

    SpotShapeType shapeType = SpotShapeType_Circular;
    SpotShapeData shapeData = {CircularSpot()};

    FlowType flowType = FlowType_None;
    FlowData flowData = {RadialFlow()};

    SimulationParametersSpotValues values;
    SimulationParametersSpotActivatedValues activatedValues;

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
        if (flowType == FlowType_Central) {
            if (flowData.centralFlow != other.flowData.centralFlow) {
                return false;
            }
        }
        if (flowType == FlowType_Linear) {
            if (flowData.linearFlow != other.flowData.linearFlow) {
                return false;
            }
        }
        if (shapeType != other.shapeType) {
            return false;
        }
        if (shapeType == SpotShapeType_Circular) {
            if (shapeData.circularSpot != other.shapeData.circularSpot) {
                return false;
            }
        }
        if (shapeType == SpotShapeType_Rectangular) {
            if (shapeData.rectangularSpot != other.shapeData.rectangularSpot) {
                return false;
            }
        }

        return (strcmp(name, other.name) == 0) && color == other.color && posX == other.posX && posY == other.posY && velX == other.velX && velY == other.velY
            && fadeoutRadius == other.fadeoutRadius && values == other.values && activatedValues == other.activatedValues && shapeType == other.shapeType;
    }
    bool operator!=(SimulationParametersSpot const& other) const { return !operator==(other); }
};
