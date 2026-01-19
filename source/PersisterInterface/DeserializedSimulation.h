#pragma once

#include <EngineInterface/Description.h>
#include <EngineInterface/StatisticsHistory.h>

#include "SettingsForSerialization.h"

struct DeserializedSimulation
{
    Desc mainData;
    SettingsForSerialization auxiliaryData;
    StatisticsHistoryData statistics;
};
