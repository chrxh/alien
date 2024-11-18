#pragma once

#include "Base/Singleton.h"
#include "EngineInterface/GenomeDescriptions.h"
#include "EngineInterface/PreviewDescriptions.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienWindow.h"
#include "Definitions.h"

class GenomeEditorWindow : public AlienWindow<SimulationFacade>
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(GenomeEditorWindow);

public:
    void openTab(GenomeDescription const& genome, bool openGenomeEditorIfClosed = true);
    GenomeDescription const& getCurrentGenome() const;

private:
    GenomeEditorWindow();

    void initIntern(SimulationFacade simulationFacade) override;
    void shutdownIntern() override;

    void processIntern() override;
    void processToolbar();
    void processEditor();

    struct TabData
    {
        int id = 0;
        GenomeDescription genome;
        std::optional<int> selectedNode;
        float previewZoom = 30.0f;
    };
    void processTab(TabData& tab);
    void processGenomeHeader(TabData& tab);
    void processConstructionSequence(TabData& tab);
    void processNode(TabData& tab, CellGenomeDescription& cell, std::optional<ShapeGeneratorResult> const& shapeGeneratorResult, bool isFirst, bool isLast);
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

    void validateAndCorrect(GenomeHeaderDescription& header) const;
    void validateAndCorrect(CellGenomeDescription& cell) const;

    void scheduleAddTab(GenomeDescription const& genome);

    void updateGeometry(GenomeDescription& genome, ConstructionShape shape);
    void setCurrentGenome(GenomeDescription const& genome);

    float _previewHeight = 0;

    mutable int _tabSequenceNumber = 0;
    std::vector<TabData> _tabDatas;
    int _selectedTabIndex = 0;
    std::optional<std::vector<uint8_t>> _copiedGenome;
    std::string _startingPath;

    //actions
    std::optional<int> _tabIndexToSelect;
    std::optional<int> _nodeIndexToJump;
    std::optional<TabData> _tabToAdd;
    std::optional<bool> _expandNodes;

    SimulationFacade _simulationFacade;
};
