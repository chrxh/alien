#pragma once

#include <cstdint>

#include "SimulationParametersTypes.h"

using Orientation = int;
enum Orientation_
{
    Orientation_Clockwise,
    Orientation_CounterClockwise
};

using FlowType = int;
enum FlowType_
{
    FlowType_None,
    FlowType_Radial,
    FlowType_Central,
    FlowType_Linear
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

union FlowAlternatives
{
    RadialFlow radialFlow;
    CentralFlow centralFlow;
    LinearFlow linearFlow;
};

struct Flow
{
    FlowType type = FlowType_None;
    FlowAlternatives alternatives = {RadialFlow()};

    bool operator==(Flow const& other) const
    {
        if (type != other.type) {
            return false;
        }
        if (type == FlowType_Radial) {
            if (alternatives.radialFlow != other.alternatives.radialFlow) {
                return false;
            }
        }
        if (type == FlowType_Central) {
            if (alternatives.centralFlow != other.alternatives.centralFlow) {
                return false;
            }
        }
        if (type == FlowType_Linear) {
            if (alternatives.linearFlow != other.alternatives.linearFlow) {
                return false;
            }
        }
        return true;
    }
};

using ZoneShapeType = int;
enum ZoneShapeType_
{
    ZoneShapeType_Circular,
    ZoneShapeType_Rectangular
};

struct CircularZone
{
    float coreRadius = 100.0f;

    bool operator==(CircularZone const&) const = default;
};

struct RectangularZone
{
    float width = 100.0f;
    float height = 200.0f;

    bool operator==(RectangularZone const&) const = default;
};

union ZoneShapeAlternatives
{
    CircularZone circularSpot;
    RectangularZone rectangularSpot;
};

struct ZoneShape
{
    ZoneShapeType type = ZoneShapeType_Circular;
    ZoneShapeAlternatives alternatives = {CircularZone()};

    bool operator==(ZoneShape const& other) const
    {
        if (type != other.type) {
            return false;
        }
        if (type == ZoneShapeType_Circular) {
            if (alternatives.circularSpot != other.alternatives.circularSpot) {
                return false;
            }
        }
        if (type == ZoneShapeType_Rectangular) {
            if (alternatives.rectangularSpot != other.alternatives.rectangularSpot) {
                return false;
            }
        }
        return true;
    }
};

struct SimulationParametersZone
{
    int locationIndex = -1;

    float posX = 0;
    float posY = 0;
    float velX= 0;
    float velY = 0;

    float fadeoutRadius = 100.0f;

    ZoneShape shape;

    Flow flow;

    bool operator==(SimulationParametersZone const&) const = default;
};
