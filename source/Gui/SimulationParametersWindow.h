#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParametersSpots.h"
#include "Definitions.h"
#include "AlienWindow.h"

class _SimulationParametersWindow : public _AlienWindow
{
public:
    _SimulationParametersWindow(SimulationController const& simController);
    ~_SimulationParametersWindow();

private:
    void processIntern() override;
    void processBackground() override;

    SimulationParametersSpot createSpot(SimulationParameters const& simParameters, int index);

    void processBase(SimulationParameters& simParameters, SimulationParameters const& origSimParameters);
    void processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot);

    SimulationController _simController;
    SimulationParametersChanger _simulationParametersChanger;

    uint32_t _savedPalette[32] = {};
    uint32_t _backupColor;
    bool _changeAutomatically = false;
};