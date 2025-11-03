#pragma once

#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "GenomeTabLayoutData.h"

class _GenomeTabWidget
{
public:
    static GenomeTabWidget createDraftTab(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeDescription const& creature,
        GenomeTabLayoutData const& layoutData = nullptr);
    static GenomeTabWidget createCreatureTab(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        uint64_t creatureId,
        GenomeDescription const& genome,
        GenomeTabLayoutData const& layoutData = nullptr);

    void process();

    void onGenomeIntoCreatureInjected();

    bool isDraft() const;
    int getTabId() const;
    std::string getName() const;

    GenomeTabEditData const& getEditData() const;
    GenomeTabLayoutData const& getLayoutData() const;
    GenomeDescription const& getGenomeDescription();

    bool hasCreaturesGenomeBeChanged() const;
    uint64_t getCreatureId();

    bool isEmpty() const;
    void convertToDraftTab();
    void resetChanges();

private:
    struct DraftData
    {};
    struct CreatureData
    {
        uint64_t creatureId;
        GenomeDescription origGenome;
        bool changesMade = false;  // true = origCreature has been changed
    };
    using SpecificEditData = std::variant<DraftData, CreatureData>;

    _GenomeTabWidget(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeDescription const& genome,
        SpecificEditData const& specificEditData,
        GenomeTabLayoutData const& layoutData = nullptr);

    void processEditors();
    void processPreview();

    void doLayout();

    // Widgets
    GenomeEditorWidget _genomeEditorWidget;
    GeneEditorWidget _geneEditorWidget;
    NodeEditorWidget _nodeEditorWidget;
    PreviewWidget _simulatedPreviewWidget;

    // Creature data
    GenomeTabEditData _editData;
    SpecificEditData _specificEditData;

    // Layout data
    GenomeTabLayoutData _origLayoutData;
    GenomeTabLayoutData _layoutData;
    std::optional<RealVector2D> _lastWindowSize;

    float _previewZoom = 30.0f;
};
