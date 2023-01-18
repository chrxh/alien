#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/PreviewDescriptions.h"

#include "AlienWindow.h"

class _GenomeEditorWindow : public _AlienWindow
{
public:
    _GenomeEditorWindow(EditorModel const& editorModel, SimulationController const& simulationController);
    ~_GenomeEditorWindow() override;

    void openTab(GenomeDescription const& genome);
    GenomeDescription const& getCurrentGenome() const;

private:
    void processIntern() override;

    void processToolbar();

    struct TabData
    {
        int id;
        GenomeDescription genome;
        std::optional<int> selectedNode;
    };
    void processTab(TabData& tab);
    void processGenomeEditTab(TabData& tab);
    void processNodeEdit(TabData& tab, CellGenomeDescription& cell);

    void showPreview(TabData& tab);

    void validationAndCorrection(CellGenomeDescription& cell) const;

    void scheduleAddTab(GenomeDescription const& genome);

    EditorModel _editorModel;
    SimulationController _simulationController;

    float _previewHeight = 200.0f;

    mutable int _tabSequenceNumber = 0;
    std::vector<TabData> _tabDatas;
    int _selectedTabIndex = 0;
    int _selectedInput = 0;
    int _selectedOutput = 0;
    float _genomeZoom = 20.0f;
    std::optional<std::vector<uint8_t>> _copiedGenome;

    //actions
    std::optional<int> _tabIndexToSelect;
    std::optional<int> _nodeIndexToJump;
    std::optional<TabData> _tabToAdd;
    bool _collapseAllNodes = false;

};
