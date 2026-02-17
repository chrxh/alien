#pragma once

#include <EngineInterface/Desc.h>
#include <EngineInterface/StatisticsHistory.h>

#include "SettingsForSerialization.h"

struct DeserializedSimulation
{
    Desc mainData;
    SettingsForSerialization auxiliaryData;
    StatisticsHistoryData statistics;
};
