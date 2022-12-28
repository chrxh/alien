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