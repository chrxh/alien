#pragma once

#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"
#include "CreatureTabLayoutData.h"

class _CreatureTabWidget
{
public:
    static CreatureTabWidget createDraftCreatureTab(
        GenomeDescription_New const& genome,
        std::optional<CreatureTabLayoutData> const& creatureTabLayoutData = std::nullopt);

    void process();

    std::string getName() const;

private:
    _CreatureTabWidget(GenomeDescription_New const& genome, std::optional<CreatureTabLayoutData> const& creatureTabLayoutData = std::nullopt);

    void processEditors();
    void processPreviews();

    void processGenomeEditor();
    void processGeneEditor();
    void processNodeEditor();
    void processDesiredConfigurationPreview();
    void processActualConfigurationPreview();

    void processGeneList();
    void processNodeList();

    void doLayout();

    int _id = 0;

    struct DraftCreature
    {
        GenomeDescription_New _genome;
    };
    struct SimulatedCreature
    {
        GenomeDescription_New _genome;
        uint64_t creatureId = 0;
    };
    std::variant<DraftCreature, SimulatedCreature> _creature;

    std::optional<int> _selectedGene;
    std::optional<int> _selectedNode;
    float _previewZoom = 30.0f;

    // Layout data
    std::optional<CreatureTabLayoutData> _creatureTabLayoutData;
    std::optional<RealVector2D> _lastWindowSize;
    std::optional<float> _lastGenomeEditorWidth;
};
