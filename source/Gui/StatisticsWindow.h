#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/RawStatisticsData.h"

#include "Definitions.h"
#include "AlienWindow.h"
#include "TimelineLiveStatistics.h"

struct ImPlotPoint;

class _StatisticsWindow : public _AlienWindow
{
public:
    _StatisticsWindow(SimulationController const& simController);
    ~_StatisticsWindow();

private:
    void processIntern() override;
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

    float calcPlotHeight(int row) const;

    SimulationController _simController;

    std::string _startingPath;

    int _plotType = 0;  //0 = accumulated, 1 = by color, 2...8 = specific color
    int _mode = 0;  //0 = real-time, 1 = entire history
    static auto constexpr MinPlotHeight = 80.0f;
    float _plotHeight = MinPlotHeight;

    std::optional<RawStatisticsData> _lastStatisticsData;
    std::optional<float> _histogramUpperBound;
    std::map<int, std::vector<double>> _cachedTimelines;
    std::unordered_set<int> _collapsedPlotIndices;

    TimelineLiveStatistics _liveStatistics;
    std::optional<std::chrono::steady_clock::time_point> _liveStatisticsDataTimepoint;
};

