#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "AlienWindow.h"

class _GenomeEditorWindow : public _AlienWindow
{
public:
    _GenomeEditorWindow();
    ~_GenomeEditorWindow() override;

private:
    void processIntern() override;

    void processToolbar();

    struct TabData
    {
        GenomeDescription genome;
        std::optional<int> selected;
    };
    void processTab(TabData& tab);
    void processGenotype(TabData& tab);
    void processCell(TabData& tab, CellGenomeDescription& cell);

    void showPhenotype(TabData& tab);

    std::optional<int> findTabToGenomeData(std::vector<uint8_t> const& genome) const;

    float _previewHeight = 200.0f;

    std::vector<TabData> _tabDatas;
    int _selectedTabIndex = 0;

    std::optional<int> _tabIndexToSelect;
    bool _collapseAllNodes = false;
    
    std::optional<std::vector<uint8_t>> _copiedGenome;
};
