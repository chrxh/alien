#pragma once

#include "EngineInterface/Definitions.h"

#include "Definitions.h"
#include "AlienWindow.h"
#include "StatisticsHistory.h"

class _StatisticsWindow : public _AlienWindow
{
public:
    _StatisticsWindow(SimulationController const& simController);

    void reset();

private:
    void processIntern();
    void processLiveStatistics();
    void processLongtermStatistics();

    void processLivePlot(int row, std::vector<float> const& valueHistory);
    void processLivePlotForCellColor(int row);
    void processLongtermPlot(int row, std::vector<float> const& valueHistory);

    void processBackground() override;

    SimulationController _simController;
    ExportStatisticsDialog _exportStatisticsDialog;

    bool _live = true;
    bool _showCellsByColor = true;

    LiveStatistics _liveStatistics;
    LongtermStatistics _longtermStatistics;
};
