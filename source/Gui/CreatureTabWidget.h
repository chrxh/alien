#pragma once

#include "EngineInterface/CreatureDescription.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"
#include "CreatureTabLayoutData.h"

class _CreatureTabWidget
{
public:
    static CreatureTabWidget createDraftCreatureTab(SimulationFacade const& simulationFacade, CreatureDescription const& genome, CreatureTabLayoutData const& layoutData = nullptr);
    static CreatureTabWidget createPinnedCreatureTab(SimulationFacade const& simulationFacade, CreatureDescription const& genome, uint64_t creatureId);

    void process();

    void onGenomeIntoCreatureInjected();

    bool isDraft() const;
    uint64_t getCreatureId() const;
    int getTabId() const;
    std::string getName() const;
    bool hasCreaturesGenomeBeChanged() const;
    CreatureDescription const& getGenome();
    bool isEmpty() const;
    void convertToDraftTab();

private:
    _CreatureTabWidget(SimulationFacade const& simulationFacade, CreatureDescription const& genome, CreatureTabLayoutData const& layoutData);
    _CreatureTabWidget(SimulationFacade const& simulationFacade, CreatureDescription const& genome, uint64_t creatureId);

    void processEditors();
    void processPreviews();

    void processPredictedPreview();
    void processSimulatedPreview();

    void doLayout();

    int _id = 0;

    // Widgets
    GenomeEditorWidget _genomeEditorWidget;
    GeneEditorWidget _geneEditorWidget;
    NodeEditorWidget _nodeEditorWidget;
    SimulatedPreviewWidget _simulatedPreviewWidget;

    // Creature data
    CreatureTabEditData _editData;
    struct PinnedCreatureData
    {
        uint64_t creatureId;
        CreatureDescription origGenome;
        bool changesMade = false;
    };
    std::optional<PinnedCreatureData> _pinnedCreatureData;

    // Layout data
    CreatureTabLayoutData _origLayoutData;
    CreatureTabLayoutData _layoutData;
    std::optional<RealVector2D> _lastWindowSize;

    float _previewZoom = 30.0f;
};
