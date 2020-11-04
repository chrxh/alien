#pragma once

#include "SimulationInfo.h"

class ParseErrorException : public std::exception
{
public:
    ParseErrorException(string const& message) : std::exception(message.c_str()) {}
};

class Parser
{
public:
    static vector<SimulationInfo> parse(QByteArray const& raw);
};
