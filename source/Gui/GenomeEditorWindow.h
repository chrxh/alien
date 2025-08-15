#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"
#include "AlienWindow.h"

class GenomeEditorWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GenomeEditorWindow);

public:
    void openTab(std::optional<uint64_t> const& creatureId, GenomeDescription const& genome, bool openEditorIfClosed = true);
    GenomeDescription getCurrentCreature() const;

private:
    GenomeEditorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processToolbar();
    void processTabWidget();

    void onInjectGenome();
    void onCreateSeed();
    void onScheduleAddCreatureTab(uint64_t creatureId, GenomeDescription const& genome);
    void onScheduleAddDraftTab(GenomeDescription const& genome);

    void pushStyleColorForTab(GenomeTabWidget const& creatureTab);

    GenomeDescription getDefaultGenome();

    SimulationFacade _simulationFacade;

    PreviewDescriptionSettings _previewSettings;
    GenomeWindowEditData _genomeEditData; 
    std::vector<GenomeTabWidget> _tabs;
    int _selectedTabIndex = 0;
    int _sequenceNumberForCreatedGenomes = 0;

    // Actions
    std::optional<GenomeTabWidget> _tabToAdd;
    std::optional<int> _tabIndexToSelect;
};
