#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/SimulationParameters.h"

#include "AlienWindow.h"

class SimulationParametersWindowPrototype : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(SimulationParametersWindowPrototype);

private:
    SimulationParametersWindowPrototype();

    void initIntern(SimulationFacade simulationFacade) override;
    void processIntern() override;

    void processToolbar();
    void processRegionMasterEditor();
    void processRegionDetailEditor();
    void processAddonList();
    void processStatusBar();

private:
    SimulationFacade _simulationFacade;

    std::optional<SimulationParameters> _copiedParameters;
};