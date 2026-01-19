#pragma once

struct SelectionShallowData
{
    int numObjects = 0;
    int numCreatures = 0;
    int numClusterCells = 0;
    int numEnergyParticles = 0;

    float centerPosX = 0;
    float centerPosY = 0;
    float centerVelX = 0;
    float centerVelY = 0;

    float clusterCenterPosX = 0;
    float clusterCenterPosY = 0;
    float clusterCenterVelX = 0;
    float clusterCenterVelY = 0;

    bool compareSizes(SelectionShallowData const& other) const
    {
        return numObjects == other.numObjects && numCreatures == other.numCreatures && numClusterCells == other.numClusterCells && numEnergyParticles == other.numEnergyParticles;
    }

    bool operator==(SelectionShallowData const& other) const = default;
};
