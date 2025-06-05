#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"
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
    void processTabWidget();

    void scheduleAddTab(GenomeDescription_New const& genome);

    SimulationFacade _simulationFacade;

    std::vector<CreatureTabWidget> _tabs;
    int _selectedTabIndex = 0;
    std::optional<CreatureTabWidget> _tabToAdd;
};
