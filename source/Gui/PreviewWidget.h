#pragma once

#include <chrono>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Description.h>
#include <EngineInterface/GenomeDescription.h>
#include <EngineInterface/GenomeDescriptionEditService.h>

#include "Definitions.h"

class _PreviewWidget
{
public:
    static PreviewWidget create(SimulationFacade const& simulationFacade, GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData);

    void process();

private:
    _PreviewWidget(SimulationFacade const& simulationFacade, GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData);

    void createSubGenomesForPreview();
    void setupPreviewData(bool useCache = true);
    void calcPreview();
    void processCreaturePreviews();
    void processCreaturePreview(int subGenomeIndex, Description&& phenotype, float width);
    void processActionBar();

    int calcTpsForPreview();

private:
    void onRun();
    void onStepBackward();
    void onStepForward();
    void onRestart();

    std::vector<uint64_t> getSeedCreatureIds() const;
    void setSeedCreatureIds(std::vector<uint64_t> const& value);

    std::vector<SubGenomeDescription> getSubGenomes() const;

    SimulationFacade _simulationFacade;
    std::vector<CreaturePreviewWidget> _creatureWidgets;

    GenomeWindowEditData _genomeEditData;
    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    bool _detailSimulation = false;
    bool _run = true;
    int _simulationSpeed = 50;  // In percent of full speed

    struct Savepoint
    {
        uint64_t timestep = 0;
        Description description;
        std::vector<uint64_t> seedCreatureIds;
    };
    std::vector<Savepoint> _savepoints;

    std::optional<GenomeDescription> _genomeFromPreviousFrame;

    std::optional<uint64_t> _previewTimestepFromPreviousMeasure;
    std::optional<std::chrono::steady_clock::time_point> _timepointFromPreviousMeasure;
    uint64_t _currentTimestep = 0;
    std::optional<int> _tpsFromPreviousMeasure;
};
