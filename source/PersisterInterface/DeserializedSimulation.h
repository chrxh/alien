#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/StatisticsHistory.h"

#include "SettingsForSerialization.h"

struct DeserializedSimulation
{
    ClusteredDataDescription mainData;
    SettingsForSerialization auxiliaryData;
    StatisticsHistoryData statistics;
};
