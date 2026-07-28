#pragma once

#include <EngineInterface/Descs.h>
#include <EngineInterface/StatisticsHistory.h>

#include "SettingsForSerialization.h"

struct DeserializedSimulation
{
    ContentDesc mainData;
    SettingsForSerialization auxiliaryData;
    StatisticsHistoryData statistics;
};
