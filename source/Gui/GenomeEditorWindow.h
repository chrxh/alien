#pragma once

#include <imgui.h>

#include <Base/Singleton.h>

#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/PreviewDesc.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienWindow.h"
#include "Definitions.h"

class GenomeEditorWindow : public AlienWindow
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GenomeEditorWindow);

public:
    void openTab(GenomeDesc const& genome, bool forceNewTab = false, bool openEditorIfClosed = true, std::optional<int> lineageId = std::nullopt);
    GenomeDesc getCurrentGenome() const;

    // Previews of the sub-genomes of the currently selected genome, e.g. for upload dialogs
    std::vector<PreviewDesc> getCurrentPreviewDescs() const;

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
    void onShareGenome();
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
    std::optional<GenomeDesc> _copiedGenome;

    // Actions
    std::vector<GenomeTabWidget> _tabsToAdd;
    std::optional<int> _tabIndexToSelect;
};
