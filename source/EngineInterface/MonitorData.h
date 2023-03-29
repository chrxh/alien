#pragma once

#include "EngineInterface/Constants.h"

struct TimestepMonitorData
{
    int numCellsByColor[MAX_COLORS] = {0, 0, 0, 0, 0, 0, 0};
    int numConnections = 0;
    int numParticles = 0;
    double internalEnergy = 0.0;

    int numCellsByAgeByColor[MAX_AGE_INTERVALS][MAX_COLORS];
};

struct TimeIntervalMonitorData
{
    float numCreatedCells = 0;
    float numSuccessfulAttacks = 0;
    float numFailedAttacks = 0;
    float numMuscleActivities = 0;
};

struct MonitorData
{
    uint64_t timestep = 0;

    TimestepMonitorData timestepData;
    TimeIntervalMonitorData timeIntervalData;
};
