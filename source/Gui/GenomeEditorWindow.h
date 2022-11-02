#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "AlienWindow.h"

class _GenomeEditorWindow : public _AlienWindow
{
public:
    _GenomeEditorWindow(EditorModel const& editorModel, SimulationController const& simulationController);
    ~_GenomeEditorWindow() override;

    void openTab(GenomeDescription const& genome);

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

    EditorModel _editorModel;
    SimulationController _simulationController;

    float _previewHeight = 200.0f;

    std::vector<TabData> _tabDatas;
    int _selectedTabIndex = 0;

    std::optional<int> _tabIndexToSelect;
    std::optional<TabData> _tabToAdd;
    bool _collapseAllNodes = false;
};
