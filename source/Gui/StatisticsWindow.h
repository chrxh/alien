#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/RawStatisticsData.h"

#include "Definitions.h"
#include "AlienWindow.h"
#include "HistogramLiveStatistics.h"
#include "TableLiveStatistics.h"
#include "TimelineLiveStatistics.h"

struct ImPlotPoint;

class _StatisticsWindow : public AlienWindow
{
public:
    _StatisticsWindow(SimulationFacade const& simulationFacade);
    ~_StatisticsWindow() override;

private:
    void processIntern() override;

    void processTimelinesTab();
    void processHistogramsTab();
    void processTablesTab();
    void processSettings();

    void processTimelineStatistics();

    void processPlot(int row, DataPoint DataPointCollection::*valuesPtr, int fracPartDecimals = 0);

    void processBackground() override;

    void plotSumColorsIntern(
        int row,
        DataPoint const* dataPoints,
        double const* timePoints,
        double const* systemClock,
        int count,
        double startTime,
        double endTime,
        int fracPartDecimals);
    void plotByColorIntern(int row, DataPoint const* values, double const* timePoints, int count, double startTime, double endTime, int fracPartDecimals);
    void plotForColorIntern(
        int row,
        DataPoint const* values,
        int colorIndex,
        double const* timePoints,
        double const* systemClock,
        int count,
        double startTime,
        double endTime,
        int fracPartDecimals);

    void setPlotScale();
    double getUpperBound(double maxValue);

    void drawValuesAtMouseCursor(
        double const* dataPoints,
        double const* timePoints,
        double const* systemClock,
        int count,
        double startTime,
        double endTime,
        double upperBound,
        int fracPartDecimals);

    void validationAndCorrection();

    float calcPlotHeight(int row) const;

    SimulationFacade _simulationFacade;

    std::string _startingPath;

    bool _settingsOpen = false;
    float _settingsHeight = 130.0f;

    using PlotScale = int;
    enum PlotScale_
    {
        PlotScale_Linear,
        PlotScale_Logarithmic,
    };
    PlotScale _plotScale = PlotScale_Linear;

    using PlotType = int;
    enum PlotType_
    {
        PlotType_Accumulated,
        PlotType_ByColor,
        PlotType_Color0,
        PlotType_Color1,
        PlotType_Color2,
        PlotType_Color3,
        PlotType_Color4,
        PlotType_Color5,
        PlotType_Color6
    };
    PlotType _plotType = PlotType_Accumulated;

    using PlotMode = int;
    enum PlotMode_
    {
        PlotMode_RealTime,
        PlotMode_EntireHistory
    };
    PlotMode _plotMode = PlotMode_RealTime;

    static auto constexpr MinPlotHeight = 80.0f;
    float _plotHeight = MinPlotHeight;

    std::optional<float> _histogramUpperBound;
    std::map<int, std::vector<double>> _cachedTimelines;
    std::unordered_set<int> _collapsedPlotIndices;

    float _timeHorizonForLiveStatistics = 10.0f;  //in seconds
    float _timeHorizonForLongtermStatistics = 100.0f;  //in percent
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
    TimelineLiveStatistics _timelineLiveStatistics;
    HistogramLiveStatistics _histogramLiveStatistics;
    TableLiveStatistics _tableLiveStatistics;
};

