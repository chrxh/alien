#pragma once

#include "EngineInterface/Definitions.h"

#include "StatisticsHistory.h"
#include "Definitions.h"

class _ExportStatisticsDialog
{
public:
    _ExportStatisticsDialog();
    ~_ExportStatisticsDialog();

    void process();

    void show(LongtermStatistics const& longtermStatistics);

private:
    void onSaveStatistics(std::string const& filename);

    std::string _startingPath;
    LongtermStatistics _statistics;
};
