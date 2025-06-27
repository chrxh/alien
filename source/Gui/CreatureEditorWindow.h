#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/CreatureDescription.h"

#include "Definitions.h"
#include "AlienWindow.h"

class CreatureEditorWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(CreatureEditorWindow);

public:
    void openTab(CreatureDescription const& creature, bool openCreatureEditorIfClosed = true);

private:
    CreatureEditorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processToolbar();
    void processTabWidget();

    void onInjectGenome();
    void onCreateSeed();
    void onScheduleAddTab(CreatureDescription const& creature, bool draft);

    void pushStyleColorForTab(CreatureTabWidget const& creatureTab);


    SimulationFacade _simulationFacade;

    std::vector<CreatureTabWidget> _tabs;
    int _selectedTabIndex = 0;

    // Actions
    std::optional<CreatureTabWidget> _tabToAdd;
    std::optional<int> _tabIndexToSelect;
};
