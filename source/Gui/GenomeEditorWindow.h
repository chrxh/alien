#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienWindow.h"
#include "Definitions.h"

class GenomeEditorWindow : public AlienWindow<>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GenomeEditorWindow);

public:
    void openTab(std::optional<uint64_t> const& creatureId, GenomeDescription const& genome, bool openEditorIfClosed = true);
    GenomeDescription getCurrentCreature() const;

private:
    GenomeEditorWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processToolbar();
    void processTabWidget();

    void onOpenGenome();
    void onSaveGenome();
    void onInjectGenome();
    void onCreateSeed();
    void onScheduleAddCreatureTab(uint64_t creatureId, GenomeDescription const& genome);
    void onScheduleAddDraftTab(GenomeDescription const& genome);

    void pushStyleColorForTab(GenomeTabWidget const& creatureTab);

    GenomeDescription getDefaultGenome();

    GenomeWindowEditData _genomeEditData;
    std::vector<GenomeTabWidget> _tabs;
    int _selectedTabIndex = 0;
    int _sequenceNumberForCreatedGenomes = 0;
    std::string _startingPath;

    // Actions
    std::optional<GenomeTabWidget> _tabToAdd;
    std::optional<int> _tabIndexToSelect;
};
