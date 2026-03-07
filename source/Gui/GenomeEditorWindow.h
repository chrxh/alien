#pragma once

#include <Base/Singleton.h>

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienWindow.h"
#include "Definitions.h"

class GenomeEditorWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GenomeEditorWindow);

public:
    void openTab(std::optional<uint64_t> const& creatureId, GenomeDesc const& genome, bool openEditorIfClosed = true);
    GenomeDesc getCurrentGenome() const;

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
    void onCloneGenome();
    void onCopyGenome();
    void onPasteGenome();
    void onInjectGenome();
    void onCreateSeed(bool provideEnergy);
    void onScheduleAddCreatureTab(uint64_t creatureId, GenomeDesc const& genome);
    void onScheduleAddDraftTab(GenomeDesc const& genome);

    void pushStyleColorForTab(GenomeTabWidget const& creatureTab);

    GenomeDesc getDefaultGenome();

    GenomeWindowEditData _genomeEditData;
    std::vector<GenomeTabWidget> _tabs;
    int _selectedTabIndex = 0;
    int _sequenceNumberForCreatedGenomes = 0;
    std::optional<int> _lastSessionId;
    std::optional<GenomeDesc> _copiedGenome;

    // Actions
    std::optional<GenomeTabWidget> _tabToAdd;
    std::optional<int> _tabIndexToSelect;
};
