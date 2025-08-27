#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescription.h"
#include "EngineInterface/GenomeDescriptionEditService.h"

#include "Definitions.h"

class _PreviewWidget
{
public:
    static PreviewWidget create(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void process();

private:
    _PreviewWidget(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void createSubGenomesForPreview();
    void setupPreviewData(bool useCache = true);
    void calcPreview();
    void processSandboxes();
    void processSandbox(int subGenomeIndex, CollectionDescription&& phenotype, float width);
    void processActionBar();

    int calcTpsForPreview();

private:
    SimulationFacade _simulationFacade;
    std::vector<CreaturePreviewWidget> _creatureWidgets;

    GenomeWindowEditData _genomeEditData;
    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    bool _run = true;
    int _simulationSpeed = 50;  // In percent of full speed
    bool _fullSimulation = false;

    std::optional<GenomeDescription> _genomeFromPreviousFrame;

    std::optional<uint64_t> _previewTimestepFromPreviousMeasure;
    std::optional<std::chrono::steady_clock::time_point> _timepointFromPreviousMeasure;
    std::optional<int> _tpsFromPreviousMeasure;
};
