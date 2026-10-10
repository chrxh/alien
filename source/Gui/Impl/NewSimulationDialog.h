#pragma once

#include <Base/Interface/Singleton.h>

#include <Data/Interface/Descs.h>
#include <Data/Interface/SimulationParameters.h>

#include <Engine/Interface/Definitions.h>

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
    Char64 _projectName = "";
    int _width = 0;
    int _height = 0;
    float _externalEnergy = 0.0f;
};
