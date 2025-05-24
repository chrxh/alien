#pragma once

#include "EngineInterface/Descriptions.h"
#include "EngineInterface/StatisticsHistory.h"

#include "SettingsForSerialization.h"

struct DeserializedSimulation
{
    CollectionDescription mainData;
    SettingsForSerialization auxiliaryData;
    StatisticsHistoryData statistics;
};
