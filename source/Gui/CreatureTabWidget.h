#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"
#include "CreatureTabLayoutData.h"

class _CreatureTabWidget
{
public:
    _CreatureTabWidget(GenomeDescription_New const& genome, std::optional<CreatureTabLayoutData> const& creatureTabLayoutData = std::nullopt);

    void process();

private:
    void processEditors();
    void processPreviews();

    void processGenomeEditor();
    void processGeneEditor();
    void processNodeEditor();
    void processDesiredConfigurationPreview();
    void processActualConfigurationPreview();

    void correctingLayout();

    int _id = 0;
    GenomeDescription_New _genome;
    std::optional<int> _selectedGene;
    std::optional<int> _selectedNode;
    float _previewZoom = 30.0f;

    // Layout data
    std::optional<CreatureTabLayoutData> _creatureTabLayoutData;
    std::optional<RealVector2D> _lastWindowSize;
    std::optional<float> _lastGenomeEditorWidth;
};
