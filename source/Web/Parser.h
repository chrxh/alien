#pragma once

#include "SimulationInfo.h"
#include "Task.h"

class ParseErrorException : public std::exception
{
public:
    ParseErrorException(string const& message) : std::exception(message.c_str()) {}
};

class Parser
{
public:
    static vector<SimulationInfo> parseForSimulationInfos(QByteArray const& raw);
    static vector<UnprocessedTask> parseForUnprocessedTasks(QByteArray const& raw);
};
