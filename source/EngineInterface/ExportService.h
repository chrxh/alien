#pragma once

#include "DataPointCollection.h"
#include "Definitions.h"

class ExportService
{
public:
    static bool exportCollectedStatistics(std::vector<DataPointCollection> const& dataPoints, std::string const& filename);
    static bool exportStatistics(uint64_t timestep, StatisticsData const& statisticsData, std::string const& filename);
};