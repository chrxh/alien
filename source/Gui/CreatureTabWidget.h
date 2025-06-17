#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"
#include "CreatureTabLayoutData.h"

class _CreatureTabWidget
{
public:
    static CreatureTabWidget createDraftCreatureTab(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData = nullptr);
    static CreatureTabWidget createRealCreatureTab(GenomeDescription_New const& genome, uint64_t creatureId);

    void process();

    bool isDraft() const;
    uint64_t getCreatureId() const;
    std::string getName() const;

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
    std::optional<uint64_t> _creatureId;

    // Layout data
    CreatureTabLayoutData _origLayoutData;
    CreatureTabLayoutData _layoutData;
    std::optional<RealVector2D> _lastWindowSize;

    float _previewZoom = 30.0f;
};
