#pragma once

#include "EngineInterface/Description.h"
#include "EngineInterface/StatisticsHistory.h"

#include "SettingsForSerialization.h"

struct DeserializedSimulation
{
    Description mainData;
    SettingsForSerialization auxiliaryData;
    StatisticsHistoryData statistics;
};
