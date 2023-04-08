#pragma once

struct RadiationSource
{
    float posX = 0;
    float posY = 0;

    bool operator==(RadiationSource const& other) const { return posX == other.posX && posY == other.posY; }
    bool operator!=(RadiationSource const& other) const { return !operator==(other); }
};