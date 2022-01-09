#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SimulationParametersSpots.h"
#include "Definitions.h"

class _SimulationParametersWindow
{
public:
    _SimulationParametersWindow(SimulationController const& simController);
    ~_SimulationParametersWindow();

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    SimulationParametersSpot createSpot(SimulationParameters const& simParameters, int index);

    void processBase(SimulationParameters& simParameters, SimulationParameters const& origSimParameters);
    void processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot);

    SimulationController _simController;

    bool _on = false;
    uint32_t _savedPalette[32] = {};
    uint32_t _backupColor;
};