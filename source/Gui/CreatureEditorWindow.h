#pragma once

#include "Base/Singleton.h"
#include "Definitions.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienWindow.h"

class CreatureEditorWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreatureEditorWindow);

public:

private:
    CreatureEditorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;

    SimulationFacade _simulationFacade;
};
