#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"
#include "AlienDialog.h"

class _NewSimulationDialog : public AlienDialog
{
public:
    _NewSimulationDialog(SimulationFacade const& simulationFacade, StatisticsWindow const& statisticsWindow);
    ~_NewSimulationDialog() override;

private:
    void processIntern() override;
    void openIntern() override;

    void onNewSimulation();

    SimulationFacade _simulationFacade;
    StatisticsWindow _statisticsWindow;

    bool _adoptSimulationParameters = true;
    int _width = 0;
    int _height = 0;
};