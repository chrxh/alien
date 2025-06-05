#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"

class _CreatureTabWidget
{
public:
    _CreatureTabWidget(CreatureTabLayoutData const& creatureTabLayoutData);

    void process();

private:

    void processEditors();
    void processPreviews();

    void processGenomeEditor();
    void processGeneEditor();
    void processNodeEditor();
    void processDesiredConfigurationPreview();
    void processActualConfigurationPreview();

    CreatureTabLayoutData _creatureTabLayoutData;

    GenomeDescription_New genome;
    std::optional<int> selectedGene;
    std::optional<int> selectedNode;
    float previewZoom = 30.0f;
};
