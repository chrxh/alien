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
    void processLiveStatistics();
    void processLongtermStatistics();

    void processHistograms();

    void processLivePlot(int colorIndex, double const* data, int fracPartDecimals = 0);
    void processLivePlotForCellsByColor(int colorIndex, ColorVector<double> const* data);
    void processLongtermPlot(int colorIndex, double const* data, int fracPartDecimals = 0);
    void processLongtermPlotForCellsByColor(int colorIndex, ColorVector<double> const* data);

    void processBackground() override;

    void plotIntern(int colorIndex, double const* data, double const* timePoints, int count, double startTime, double endTime, int fracPartDecimals);
    void plotByColorIntern(int colorIndex, ColorVector<double> const* data, double const* timePoints, int count, double startTime, double endTime);

    SimulationController _simController;
    ExportStatisticsDialog _exportStatisticsDialog;

    bool _live = true;
    bool _showCellsByColor = false;

    std::optional<StatisticsData> _lastStatisticsData;
    std::optional<float> _histogramUpperBound;

    TimelineLiveStatistics _liveStatistics;
    TimelineLongtermStatistics _longtermStatistics;
};
