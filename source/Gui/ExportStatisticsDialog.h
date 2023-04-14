#pragma once

#include "CollectedStatisticsData.h"
#include "Definitions.h"

class _ExportStatisticsDialog
{
public:
    _ExportStatisticsDialog();
    ~_ExportStatisticsDialog();

    void process();

    void show(TimelineLongtermStatistics const& longtermStatistics);

private:
    void onSaveStatistics(std::string const& filename);

    std::string _startingPath;
    TimelineLongtermStatistics _statistics;
};
