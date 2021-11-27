#pragma once

struct SelectionShallowData
{
    int numCells = 0;
    int numClusterCells = 0;
    int numParticles = 0;

    float centerPosX = 0;
    float centerPosY = 0;
    float centerVelX = 0;
    float centerVelY = 0;

    float extCenterPosX = 0;
    float extCenterPosY = 0;
    float extCenterVelX = 0;
    float extCenterVelY = 0;

    bool operator==(SelectionShallowData const& other) const
    {
        return numCells == other.numCells && numClusterCells == other.numClusterCells
            && numParticles == other.numParticles && centerPosX == other.centerPosX && centerPosY == other.centerPosY
            && centerVelX == other.centerVelX && centerVelY == other.centerVelY && extCenterPosX == other.extCenterPosX
            && extCenterPosY == other.extCenterPosY && extCenterVelX == other.extCenterVelX
            && extCenterVelY == other.extCenterVelY;
    }
    bool operator!=(SelectionShallowData const& other) const { return !(*this == other); }
};