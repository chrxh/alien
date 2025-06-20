#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"
#include "CreatureTabLayoutData.h"

class _CreatureTabWidget
{
public:
    static CreatureTabWidget createDraftCreatureTab(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData = nullptr);
    static CreatureTabWidget createPinnedCreatureTab(GenomeDescription_New const& genome, uint64_t creatureId);

    void process();

    void onGenomeIntoCreatureInjected();

    bool isDraft() const;
    uint64_t getCreatureId() const;
    int getTabId() const;
    std::string getName() const;
    bool hasCreaturesGenomeBeChanged() const;
    GenomeDescription_New const& getGenome();
    bool isEmpty() const;

private:
    _CreatureTabWidget(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData);
    _CreatureTabWidget(GenomeDescription_New const& genome, uint64_t creatureId);

    void processEditors();
    void processPreviews();

    void processDesiredConfigurationPreview();
    void processActualConfigurationPreview();

    void doLayout();

    int _id = 0;

    // Widgets
    GenomeEditorWidget _genomeEditorWidget;
    GeneEditorWidget _geneEditorWidget;
    NodeEditorWidget _nodeEditorWidget;

    // Creature data
    CreatureTabEditData _editData;
    struct PinnedCreatureData
    {
        uint64_t creatureId;
        GenomeDescription_New origGenome;
        bool changesMade = false;
    };
    std::optional<PinnedCreatureData> _pinnedCreatureData;

    // Layout data
    CreatureTabLayoutData _origLayoutData;
    CreatureTabLayoutData _layoutData;
    std::optional<RealVector2D> _lastWindowSize;

    float _previewZoom = 30.0f;
};
