#pragma once

#include "EngineInterface/SimulationParameters.h"
#include "EngineInterface/SimulationParametersSpots.h"
#include "EngineImpl/Definitions.h"
#include "Definitions.h"

class _SimulationParametersWindow
{
public:
    _SimulationParametersWindow(StyleRepository const& styleRepository, SimulationController const& simController);

    void process();

    bool isOn() const;
    void setOn(bool value);

private:
    SimulationParametersSpot createSpot(SimulationParameters const& simParameters, int index);

    void processBase(SimulationParameters& simParameters, SimulationParameters const& origSimParameters);
    void processSpot(SimulationParametersSpot& spot, SimulationParametersSpot const& origSpot);

    void createGroup(std::string const& name);

    StyleRepository _styleRepository;
    SimulationController _simController;

    bool _on = false;
    uint32_t _savedPalette[32] = {};
    uint32_t _backupColor;
};