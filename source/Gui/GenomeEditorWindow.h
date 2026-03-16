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
    void openTab(GenomeDesc const& genome, bool forceNewTab = false, bool openEditorIfClosed = true);
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
    void onSavepointGenome();
    void onInjectGenome();
    void onCreateSeed(bool provideEnergy);
    void onScheduleAddTab(GenomeDesc const& genome);

    void pushStyleColorForTab(GenomeTabWidget const& genomeTab);

    GenomeDesc getDefaultGenome();

    GenomeWindowEditData _genomeEditData;
    std::vector<GenomeTabWidget> _tabs;
    int _selectedTabIndex = 0;
    int _sequenceNumberForCreatedGenomes = 0;
    std::optional<GenomeDesc> _copiedGenome;

    // Actions
    std::vector<GenomeTabWidget> _tabsToAdd;
    std::optional<int> _tabIndexToSelect;
};
