#pragma once

struct MonitorData
{
    uint64_t timeStep = 0;

    //entities
    int numCellsByColor[7] = {0, 0, 0, 0, 0, 0, 0};
    int numParticles = 0;
    int numTokens = 0;
    double totalInternalEnergy = 0.0;

    //processes
    int numCreatedCells = 0;
    int numSuccessfulAttacks = 0;
    int numFailedAttacks = 0;
    int numMuscleActivities = 0;
};
