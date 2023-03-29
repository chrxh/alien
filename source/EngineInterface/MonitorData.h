#pragma once

#include "EngineInterface/Constants.h"

struct MonitorData
{
    uint64_t timestep = 0;

    //time step data
    int numCellsByColor[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    int numConnections = 0;
    int numParticles = 0;
    double totalInternalEnergy = 0.0;

    //time interval data
    float numCreatedCells = 0;
    float numSuccessfulAttacks = 0;
    float numFailedAttacks = 0;
    float numMuscleActivities = 0;
};
