#pragma once

#include <imgui.h>

#include <Base/Singleton.h>

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienWindow.h"
#include "Definitions.h"

class GenomeEditorWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GenomeEditorWindow);

public:
    // lineageId marks a genome that was opened from an inspected creature; it colors the tab marker
    void openTab(GenomeDesc const& genome, bool forceNewTab = false, bool openEditorIfClosed = true, std::optional<int> lineageId = std::nullopt);
    GenomeDesc getCurrentGenome() const;

private:
    GenomeEditorWindow();

    void initIntern() override;
    void shutdownIntern() override;
    void processIntern() override;
    bool isShown() override;

    void processToolbar();
    void processUnsavedChangesChip(bool hasGenomeChanged);
    void processTabWidget();
    std::string getTabLabel(GenomeTabWidget const& genomeTab);
    void processLineageMarker(GenomeTabWidget const& genomeTab, ImGuiID tabId);

    void onOpenGenome();
    void onSaveGenome();
    void onCloneGenome();
    void onCloseOtherTabs();
    void onCopyGenome();
    void onPasteGenome();
    void onSavepointGenome();
    void onInjectGenome();
    void onCreateSeed(bool provideEnergy);
    void onScheduleAddTab(GenomeDesc const& genome, std::optional<int> lineageId);

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
