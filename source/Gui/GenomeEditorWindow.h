#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/PreviewDescriptions.h"

#include "AlienWindow.h"

class _GenomeEditorWindow : public _AlienWindow
{
public:
    _GenomeEditorWindow(EditorModel const& editorModel, SimulationController const& simulationController, Viewport const& viewport);
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
    template<typename Description>
    void processSubGenomeWidgets(TabData const& tab, Description& desc);

    void onOpenGenome();
    void onSaveGenome();
    void onAddNode();
    void onDeleteNode();
    void onNodeDecreaseSequenceNumber();
    void onNodeIncreaseSequenceNumber();

    void showPreview(TabData& tab);

    void validationAndCorrection(CellGenomeDescription& cell) const;

    void scheduleAddTab(GenomeDescription const& genome);
    void onCreateSpore();

    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;

    float _previewHeight = 200.0f;

    mutable int _tabSequenceNumber = 0;
    std::vector<TabData> _tabDatas;
    int _selectedTabIndex = 0;
    int _selectedInput = 0;
    int _selectedOutput = 0;
    float _genomeZoom = 20.0f;
    std::optional<std::vector<uint8_t>> _copiedGenome;
    std::string _startingPath;

    //actions
    std::optional<int> _tabIndexToSelect;
    std::optional<int> _nodeIndexToJump;
    std::optional<TabData> _tabToAdd;
    bool _collapseAllNodes = false;

};
