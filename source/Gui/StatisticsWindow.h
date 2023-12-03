#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/RawStatisticsData.h"

#include "Definitions.h"
#include "AlienWindow.h"
#include "CollectedStatisticsData.h"

struct ImPlotPoint;

class _StatisticsWindow : public _AlienWindow
{
public:
    _StatisticsWindow(SimulationController const& simController);
    ~_StatisticsWindow();

    void reset();

private:
    void processIntern();
    void processTimelines();
    void processTimelineStatistics();

    void processHistograms();

    void processPlot(int row, DataPoint DataPointCollection::*valuesPtr, int fracPartDecimals = 0);

    void processBackground() override;

    void plotSumColorsIntern(int row, DataPoint const* dataPoint, double const* timePoints, int count, double startTime, double endTime, int fracPartDecimals);
    void plotByColorIntern(int row, DataPoint const* values, double const* timePoints, int count, double startTime, double endTime, int fracPartDecimals);
    void plotForColorIntern(
        int row,
        DataPoint const* values,
        int colorIndex,
        double const* timePoints,
        int count,
        double startTime,
        double endTime,
        int fracPartDecimals);

    void onSaveStatistics();

    float getPlotHeight() const;

    SimulationController _simController;

    bool _live = true;
    std::string _startingPath;
    int _plotType = 0;
    bool _maximize = false;

    std::optional<RawStatisticsData> _lastStatisticsData;
    std::optional<float> _histogramUpperBound;
    std::map<int, std::vector<double>> _cachedTimelines;

    TimelineLiveStatistics _liveStatistics;
//    TimelineLongtermStatistics _longtermStatistics;
};
