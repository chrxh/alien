#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/StatisticsHistory.h"

#include "SettingsForSerialization.h"

struct DeserializedSimulation
{
    DataDescription mainData;
    SettingsForSerialization auxiliaryData;
    StatisticsHistoryData statistics;
};
