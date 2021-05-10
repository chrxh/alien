#pragma once

#include "SimulationInfo.h"
#include "Task.h"

class Parser
{
public:
    static vector<SimulationInfo> parseForSimulationInfos(QByteArray const& raw);
    static vector<Task> parseForUnprocessedTasks(QByteArray const& raw);
};
