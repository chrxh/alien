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
    bool isShown() override;

    void processToolbar();
    void processEditors();
    void processPreviews();

    SimulationFacade _simulationFacade;

    float _genomeEditorWidth = 200.0f;
    float _previewsHeight = 300.0f;
};
