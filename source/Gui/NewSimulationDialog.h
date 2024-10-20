#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "Definitions.h"
#include "AlienDialog.h"

class NewSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NewSimulationDialog);

public:
    void init(SimulationFacade const& simulationFacade);
    void shutdown();

private:
    NewSimulationDialog();
    void processIntern() override;
    void openIntern() override;

    void onNewSimulation();

    SimulationFacade _simulationFacade;

    bool _adoptSimulationParameters = true;
    int _width = 0;
    int _height = 0;
};