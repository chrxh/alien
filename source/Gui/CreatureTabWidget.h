#pragma once

#include "EngineInterface/CreatureDescription.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"
#include "CreatureTabLayoutData.h"

class _CreatureTabWidget
{
public:
    static CreatureTabWidget createDraftCreatureTab(SimulationFacade const& simulationFacade, CreatureDescription const& creature, CreatureTabLayoutData const& layoutData = nullptr);
    static CreatureTabWidget createSimulatedCreatureTab(SimulationFacade const& simulationFacade, CreatureDescription const& creature);

    void process();

    void onGenomeIntoCreatureInjected();

    bool isDraft() const;
    int getTabId() const;
    std::string getName() const;
    bool hasCreaturesGenomeBeChanged() const;
    CreatureDescription const& getCreatureDescription();
    bool isEmpty() const;
    void convertToDraftTab();

private:
    _CreatureTabWidget(SimulationFacade const& simulationFacade, CreatureDescription const& genome, bool draft, CreatureTabLayoutData const& layoutData = nullptr);

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
    struct DraftCreatureData
    {
    };
    struct SimulatedCreatureData
    {
        CreatureDescription origCreature;
        bool changesMade = false;
    };
    using SpecificEditData = std::variant<DraftCreatureData, SimulatedCreatureData>;
    SpecificEditData _specificEditData;

    // Layout data
    CreatureTabLayoutData _origLayoutData;
    CreatureTabLayoutData _layoutData;
    std::optional<RealVector2D> _lastWindowSize;

    float _previewZoom = 30.0f;
};
