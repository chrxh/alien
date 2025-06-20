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
    void openTab(GenomeDescription_New const& genome, uint64_t creatureId, bool openCreatureEditorIfClosed = true);

private:
    CreatureEditorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processToolbar();
    void processTabWidget();

    void onInjectGenome();
    void onScheduleAddTab(GenomeDescription_New const& genome, std::optional<uint64_t> const& creatureId = std::nullopt);

    void pushStyleColorForTab(CreatureTabWidget const& creatureTab);


    SimulationFacade _simulationFacade;

    std::vector<CreatureTabWidget> _tabs;
    int _selectedTabIndex = 0;

    // Actions
    std::optional<CreatureTabWidget> _tabToAdd;
    std::optional<int> _tabIndexToSelect;
};
