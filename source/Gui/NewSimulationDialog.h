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

    ~_NewSimulationDialog();

    void process();

    void show();

private:
    void onNewSimulation();

    SimulationController _simController;
    Viewport _viewport;
    StatisticsWindow _statisticsWindow;

    bool _on = false;
    bool _adoptSimulationParameters = true;
    bool _adoptSymbols = true;
    int _width = 0;
    int _height = 0;
};