#pragma once

enum class Orientation
{
    Clockwise,
    CounterClockwise,
};

struct RadialFlowCenterData
{
    float posX = 0;
    float posY = 0;
    float radius = 200.0f;
    float strength = 0.01f;
    Orientation orientation = Orientation::Clockwise;

    bool operator==(RadialFlowCenterData const& other) const
    {
        return posX == other.posX && posY == other.posY && radius == other.radius && strength == other.strength
            && orientation == other.orientation;
    }

    bool operator!=(RadialFlowCenterData const& other) const { return !operator==(other); }
};

struct FlowFieldSettings
{
    bool active = false;

    int numCenters = 1; //only 2 centers supported
    RadialFlowCenterData radialFlowCenters[2];

    bool operator==(FlowFieldSettings const& other) const
    {
        return active == other.active && numCenters == other.numCenters && radialFlowCenters == other.radialFlowCenters;
    }

    bool operator!=(FlowFieldSettings const& other) const { return !operator==(other); }
};
