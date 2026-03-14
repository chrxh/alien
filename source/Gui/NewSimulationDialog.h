#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/SimulationParameters.h>

#include "AlienDialog.h"
#include "Definitions.h"

class NewSimulationDialog : public AlienDialog
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(NewSimulationDialog);

private:
    NewSimulationDialog();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    void openIntern() override;

    void onNewSimulation();

    bool _adoptSimulationParameters = true;
    Char64 _projectName = "<unnamed simulation>";
    int _width = 0;
    int _height = 0;
};
