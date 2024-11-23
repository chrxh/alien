#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/StatisticsHistory.h"

#include "AuxiliaryData.h"

struct DeserializedSimulation
{
    ClusteredDataDescription mainData;
    AuxiliaryData auxiliaryData;
    StatisticsHistoryData statistics;
};
