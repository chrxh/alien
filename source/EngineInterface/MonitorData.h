#pragma once

struct MonitorData
{
    int timeStep = 0;
    int numClusters = 0;
    int numClustersWithTokens = 0;
    int numCells = 0;
    int numParticles = 0;
    int numTokens = 0;
    double totalInternalEnergy = 0.0;
    double totalLinearKineticEnergy = 0.0;
    double totalRotationalKineticEnergy = 0.0;
};
