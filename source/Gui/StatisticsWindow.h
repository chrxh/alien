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

    void processPlot(int colorIndex, ColorVector<double> DataPoint::*valuesPtr, int fracPartDecimals = 0);

    void processBackground() override;

    void
    plotIntern(int colorIndex, ColorVector<double> const* values, double const* timePoints, int count, double startTime, double endTime, int fracPartDecimals);
    void plotByColorIntern(int colorIndex, ColorVector<double> const* values, double const* timePoints, int count, double startTime, double endTime);

    SimulationController _simController;
    ExportStatisticsDialog _exportStatisticsDialog;

    bool _live = true;
    int _plotType = 0;

    std::optional<StatisticsData> _lastStatisticsData;
    std::optional<float> _histogramUpperBound;
    std::map<int, std::vector<double>> _cachedTimelines;

    TimelineLiveStatistics _liveStatistics;
    TimelineLongtermStatistics _longtermStatistics;
};
