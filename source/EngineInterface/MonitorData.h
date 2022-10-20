#pragma once

struct MonitorData
{
    uint64_t timestep = 0;

    //entities
    int numCellsByColor[7] = {0, 0, 0, 0, 0, 0, 0};
    int numConnections = 0;
    int numParticles = 0;
    double totalInternalEnergy = 0.0;

    //processes
    float numCreatedCells = 0;
    float numSuccessfulAttacks = 0;
    float numFailedAttacks = 0;
    float numMuscleActivities = 0;
};
