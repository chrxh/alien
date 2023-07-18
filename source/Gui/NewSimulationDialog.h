#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"
#include "AlienDialog.h"

class _NewSimulationDialog : public _AlienDialog
{
public:
    _NewSimulationDialog(
        SimulationController const& simController,
        TemporalControlWindow const& temporalControlWindow,
        Viewport const& viewport,
        StatisticsWindow const& statisticsWindow);

    ~_NewSimulationDialog();

private:
    void processIntern() override;
    void openIntern() override;

    void onNewSimulation();

    SimulationController _simController;
    TemporalControlWindow _temporalControlWindow;
    Viewport _viewport;
    StatisticsWindow _statisticsWindow;

    bool _adoptSimulationParameters = true;
    int _width = 0;
    int _height = 0;
};