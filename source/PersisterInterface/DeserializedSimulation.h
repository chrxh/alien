#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/StatisticsHistory.h"

#include "SettingsData.h"

struct DeserializedSimulation
{
    ClusteredDataDescription mainData;
    SettingsData auxiliaryData;
    StatisticsHistoryData statistics;
};
