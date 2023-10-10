#pragma once

#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/PreviewDescriptions.h"

#include "AlienWindow.h"
#include "Definitions.h"

class _GenomeEditorWindow : public _AlienWindow
{
public:
    _GenomeEditorWindow(EditorModel const& editorModel, SimulationController const& simulationController, Viewport const& viewport);
    ~_GenomeEditorWindow() override;

    void registerCyclicReferences(UploadSimulationDialogWeakPtr const& uploadSimulationDialog);

    void openTab(GenomeDescription const& genome, bool openGenomeEditorIfClosed = true);
    GenomeDescription const& getCurrentGenome() const;

private:
    void processIntern() override;
    void processToolbar();
    void processEditor();

    struct TabData
    {
        int id;
        GenomeDescription genome;
        std::optional<int> selectedNode;
    };
    void processTab(TabData& tab);
    void processGenomeHeader(TabData& tab);
    void processConstructionSequence(TabData& tab);
    void processNode(TabData& tab, CellGenomeDescription& cell, std::optional<ShapeGeneratorResult> const& shapeGeneratorResult, bool isFirstOrLast);
    template<typename Description>
    void processSubGenomeWidgets(TabData const& tab, Description& desc);

    void onOpenGenome();
    void onSaveGenome();
    void onUploadGenome();
    void onAddNode();
    void onDeleteNode();
    void onNodeDecreaseSequenceNumber();
    void onNodeIncreaseSequenceNumber();
    void onCreateSpore();

    void showPreview(TabData& tab);

    void validationAndCorrection(GenomeHeaderDescription& info) const;
    void validationAndCorrection(CellGenomeDescription& cell) const;

    void scheduleAddTab(GenomeDescription const& genome);

    void updateGeometry(GenomeDescription& genome, ConstructionShape shape);

    float _previewHeight = 0;

    mutable int _tabSequenceNumber = 0;
    std::vector<TabData> _tabDatas;
    int _selectedTabIndex = 0;
    int _selectedInput = 0;
    int _selectedOutput = 0;
    float _previewZoom = 30.0f;
    std::optional<std::vector<uint8_t>> _copiedGenome;
    std::string _startingPath;

    //actions
    std::optional<int> _tabIndexToSelect;
    std::optional<int> _nodeIndexToJump;
    std::optional<TabData> _tabToAdd;
    std::optional<bool> _expandNodes;

    EditorModel _editorModel;
    SimulationController _simController;
    Viewport _viewport;
    UploadSimulationDialogWeakPtr _uploadSimulationDialog;
};
