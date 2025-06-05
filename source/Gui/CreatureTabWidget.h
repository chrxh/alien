#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"

class _CreatureTabWidget
{
public:
    _CreatureTabWidget(GenomeDescription_New const& genome, CreatureTabLayoutData const& creatureTabLayoutData);

    void process();

private:

    void processEditors();
    void processPreviews();

    void processGenomeEditor();
    void processGeneEditor();
    void processNodeEditor();
    void processDesiredConfigurationPreview();
    void processActualConfigurationPreview();

    int _id = 0;
    GenomeDescription_New _genome;
    std::optional<int> _selectedGene;
    std::optional<int> _selectedNode;
    float _previewZoom = 30.0f;

    CreatureTabLayoutData _creatureTabLayoutData;
};
