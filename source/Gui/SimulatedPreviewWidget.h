#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescription.h"
#include "EngineInterface/GenomeDescriptionEditService.h"

#include "Definitions.h"

class _SimulatedPreviewWidget
{
public:
    static SimulatedPreviewWidget create(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void process();

private:
    _SimulatedPreviewWidget(
        SimulationFacade const& simulationFacade,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void createSubGenomesForPreview();
    void setupPreviewData(bool useCache = true);
    void calcPreview();
    void processSandboxes();
    void processSandbox(int subGenomeIndex, CollectionDescription&& phenotype, int geneStartIndex, float width);
    void processActionBar();

    int calcTpsForPreview();

private:
    SimulationFacade _simulationFacade;
    std::vector<PreviewDescriptionWidget> _previewWidgets;

    GenomeWindowEditData _genomeEditData;
    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    bool _run = true;
    int _simulationSpeed = 50;  // In percent of full speed
    bool _fullSimulation = false;

    std::vector<GeneIndicesForSubGenome> _geneIndicesForSubGenomes;
    std::vector<GenomeDescriptionWithStartGeneIndex> _subGenomesForPreview;
    std::vector<uint64_t> _seedCreatureIdsForPreview;

    std::optional<GenomeDescription> _genomeFromPreviousFrame;

    std::optional<uint64_t> _previewTimestepFromPreviousMeasure;
    std::optional<std::chrono::steady_clock::time_point> _timepointFromPreviousMeasure;
    std::optional<int> _tpsFromPreviousMeasure;
};
