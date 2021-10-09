#pragma once

#include "EngineImpl/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"

class _NewSimulationDialog
{
public:
    _NewSimulationDialog(
        SimulationController const& simController,
        Viewport const& viewport,
        StatisticsWindow const& statisticsWindow,
        StyleRepository const& styleRepository);

    void process();

    void show();

private:
    void onNewSimulation();

    SimulationController _simController;
    Viewport _viewport;
    StatisticsWindow _statisticsWindow;
    StyleRepository _styleRepository;

    bool _on = false;
    int _width = 0;
    int _height = 0;
};