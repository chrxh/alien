#pragma once

#include "AuxiliaryData.h"
#include "Descriptions.h"
#include "StatisticsHistory.h"

struct DeserializedSimulation
{
    ClusteredDataDescription mainData;
    AuxiliaryData auxiliaryData;
    StatisticsHistoryData statistics;
};
