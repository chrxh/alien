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

    float clusterCenterPosX = 0;
    float clusterCenterPosY = 0;
    float clusterCenterVelX = 0;
    float clusterCenterVelY = 0;

    bool compareNumbers(SelectionShallowData const& other) const
    {
        return numCells == other.numCells && numClusterCells == other.numClusterCells && numParticles == other.numParticles;
    }

    bool operator==(SelectionShallowData const& other) const
    {
        return numCells == other.numCells && numClusterCells == other.numClusterCells
            && numParticles == other.numParticles && centerPosX == other.centerPosX && centerPosY == other.centerPosY
            && centerVelX == other.centerVelX && centerVelY == other.centerVelY && clusterCenterPosX == other.clusterCenterPosX
            && clusterCenterPosY == other.clusterCenterPosY && clusterCenterVelX == other.clusterCenterVelX
            && clusterCenterVelY == other.clusterCenterVelY;
    }
    bool operator!=(SelectionShallowData const& other) const { return !(*this == other); }
};
