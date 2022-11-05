#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "AlienWindow.h"
#include "PreviewDescriptions.h"

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
        std::optional<int> selectedNode;
    };
    void processTab(TabData& tab);
    void processGenomeEditTab(TabData& tab);
    void processNodeEdit(TabData& tab, CellGenomeDescription& cell);

    void showPreview(TabData& tab);

    void validationAndCorrection(CellGenomeDescription& cell) const;

    EditorModel _editorModel;
    SimulationController _simulationController;

    float _previewHeight = 200.0f;

    std::vector<TabData> _tabDatas;
    int _selectedTabIndex = 0;

    //actions
    std::optional<int> _tabIndexToSelect;
    std::optional<TabData> _tabToAdd;
    bool _collapseAllNodes = false;

};
