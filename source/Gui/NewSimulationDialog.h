#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

class _NewSimulationDialog
{
public:
    _NewSimulationDialog(
        SimulationController const& simController,
        Viewport const& viewport,
        StatisticsWindow const& statisticsWindow);

    void process();

    void show();

private:
    void onNewSimulation();

    SimulationController _simController;
    Viewport _viewport;
    StatisticsWindow _statisticsWindow;

    bool _on = false;
    int _width = 0;
    int _height = 0;
};