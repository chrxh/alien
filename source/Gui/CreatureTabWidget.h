#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"
#include "CreatureTabLayoutData.h"

class _CreatureTabWidget
{
public:
    static CreatureTabWidget createDraftCreatureTab(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData = nullptr);

    void process();

    std::string getName() const;

private:
    _CreatureTabWidget(GenomeDescription_New const& genome, CreatureTabLayoutData const& layoutData);

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
