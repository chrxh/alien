#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParametersSpot.h"
#include "Definitions.h"
#include "AlienWindow.h"

class _SimulationParametersWindow : public _AlienWindow
{
public:
    _SimulationParametersWindow(SimulationController const& simController, RadiationSourcesWindow const& radiationSourcesWindow);
    ~_SimulationParametersWindow();

private:
    void processIntern() override;
    void processBackground() override;

    SimulationParametersSpot createSpot(SimulationParameters const& simParameters, int index);
    void createDefaultSpotData(SimulationParametersSpot& spot);

    void processBase(SimulationParameters& simParameters, SimulationParameters const& origSimParameters);
    void processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot, SimulationParameters const& parameters);

    void validationAndCorrection(SimulationParameters& parameters) const;
    void validationAndCorrection(SimulationParametersSpot& spot) const;

    SimulationController _simController;
    RadiationSourcesWindow _radiationSourcesWindow;
    SimulationParametersChanger _simulationParametersChanger;

    uint32_t _savedPalette[32] = {};
    uint32_t _backupColor;
    bool _changeAutomatically = false;
};