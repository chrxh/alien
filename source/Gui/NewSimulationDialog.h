#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"
#include "AlienDialog.h"

class NewSimulationDialog : public AlienDialog<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NewSimulationDialog);

private:
    NewSimulationDialog();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern() override;

    void onNewSimulation();

    SimulationFacade _simulationFacade;

    bool _adoptSimulationParameters = true;
    int _width = 0;
    int _height = 0;
};