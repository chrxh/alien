#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/StatisticsData.h"

#include "Definitions.h"
#include "AlienWindow.h"
#include "CollectedStatisticsData.h"

struct ImPlotPoint;

class _StatisticsWindow : public _AlienWindow
{
public:
    _StatisticsWindow(SimulationController const& simController);

    void reset();

private:
    void processIntern();
    void processTimelines();
    void processTimelineStatistics();

    void processHistograms();

    void processPlot(int row, ColorVector<double> DataPoint::*valuesPtr, double const DataPoint::*summedValuesPtr = nullptr, int fracPartDecimals = 0);

    void processBackground() override;

    void plotSumColorsIntern(
        int row,
        ColorVector<double> const* values,
        double const* summedValues,
        double const* timePoints,
        int count,
        double startTime,
        double endTime,
        int fracPartDecimals);
    void
    plotByColorIntern(
        int row,
        ColorVector<double> const* values,
        double const* timePoints,
        int count,
        double startTime,
        double endTime,
        int fracPartDecimals);
    void plotForColorIntern(
        int row,
        ColorVector<double> const* values,
        int colorIndex,
        double const* timePoints,
        int count,
        double startTime,
        double endTime,
        int fracPartDecimals);

    void onSaveStatistics(std::string const& filename);

    SimulationController _simController;

    bool _live = true;
    std::string _startingPath;
    int _plotType = 0;

    std::optional<StatisticsData> _lastStatisticsData;
    std::optional<float> _histogramUpperBound;
    std::map<int, std::vector<double>> _cachedTimelines;

    TimelineLiveStatistics _liveStatistics;
    TimelineLongtermStatistics _longtermStatistics;
};
