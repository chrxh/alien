#pragma once

struct ParticleSource
{
    float posX = 0;
    float posY = 0;

    bool operator==(ParticleSource const& other) const { return posX == other.posX && posY == other.posY; }
    bool operator!=(ParticleSource const& other) const { return !operator==(other); }
};