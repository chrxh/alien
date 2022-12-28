#pragma once

enum class Orientation
{
    Clockwise,
    CounterClockwise,
};

struct FlowCenter
{
    float posX = 0;
    float posY = 0;
    float radius = 200.0f;
    float strength = 0.001f;
    Orientation orientation = Orientation::Clockwise;

    bool operator==(FlowCenter const& other) const
    {
        return posX == other.posX && posY == other.posY && radius == other.radius && strength == other.strength
            && orientation == other.orientation;
    }
    bool operator!=(FlowCenter const& other) const { return !operator==(other); }
};

struct FlowFieldSettings
{
    int numCenters = 0; //only 2 centers supported
    FlowCenter centers[2];

    bool operator==(FlowFieldSettings const& other) const
    {
        if (numCenters != other.numCenters) {
            return false;
        }
        for (int i = 0; i < 2; ++i) {
            if (centers[i] != other.centers[i]) {
                return false;
            }
        }
        return true;
    }
    bool operator!=(FlowFieldSettings const& other) const { return !operator==(other); }
};
