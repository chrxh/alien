#pragma once

#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "GenomeTabLayoutData.h"

class _GenomeTabWidget
{
public:
    static GenomeTabWidget createDraftTab(
        GenomeWindowEditData const& genomeEditData,
        GenomeDesc const& creature,
        GenomeTabLayoutData const& layoutData = nullptr);
    static GenomeTabWidget createCreatureTab(
        GenomeWindowEditData const& genomeEditData,
        uint64_t creatureId,
        GenomeDesc const& genome,
        GenomeTabLayoutData const& layoutData = nullptr);

    void process();

    void onGenomeIntoCreatureInjected();

    bool isDraft() const;
    int getTabId() const;
    std::string getName() const;

    GenomeTabEditData const& getEditData() const;
    GenomeTabLayoutData const& getLayoutData() const;
    GenomeDesc const& getGenomeDesc() const;

    void setGenomeDesc(GenomeDesc const& genome);

    bool hasCreaturesGenomeBeChanged() const;
    uint64_t getCreatureId();

    bool isEmpty() const;
    void convertToDraftTab();
    void resetOriginal();
    void resetChanges();

private:
    struct DraftData
    {};
    struct CreatureData
    {
        uint64_t creatureId = 0;
        GenomeDesc origGenome;
        bool changesMade = false;  // true = origCreature has been changed
    };
    using SpecificEditData = std::variant<DraftData, CreatureData>;

    _GenomeTabWidget(
        GenomeWindowEditData const& genomeEditData,
        GenomeDesc const& genome,
        SpecificEditData const& specificEditData,
        GenomeTabLayoutData const& layoutData = nullptr);

    void processEditors();
    void processPreview();

    void doLayout();

    void updateSpecificEditDataFromSimulation();

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
