#pragma once

#include <chrono>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/Desc.h>
#include <EngineInterface/GenomeDesc.h>
#include <EngineInterface/GenomeDescEditService.h>

#include "Definitions.h"

class _PreviewWidget
{
public:
    static PreviewWidget create(GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData);

    void process();

private:
    _PreviewWidget(GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData);

    void createSubGenomesForPreview();
    void setupPreviewData(bool useCache = true);
    void calcPreview();
    void processCreaturePreviews();
    void processCreaturePreview(bool& phenotypeChanged, int subGenomeIndex, Desc& phenotype, float width);
    void processActionBar();

    int calcTpsForPreview();

private:
    void onRun();
    void onStepBackward();
    void onStepForward();
    void onRestart();

    std::vector<uint64_t> getSeedCreatureIds() const;
    void setSeedCreatureIds(std::vector<uint64_t> const& value);

    std::vector<SubGenomeDesc> getSubGenomes() const;

    std::vector<CreaturePreviewWidget> _creatureWidgets;

    GenomeWindowEditData _genomeEditData;
    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    struct Savepoint
    {
        uint64_t timestep = 0;
        Desc description;
        std::vector<uint64_t> seedCreatureIds;
    };
    std::vector<Savepoint> _savepoints;

    std::optional<GenomeDesc> _genomeFromPreviousFrame;
    std::optional<GenomeDesc> _genomeForCurrentPreview;  // Can be different from _editData->genome when simulation is paused
    std::optional<uint64_t> _sessionIdFromPreviousFrame;

    std::optional<uint64_t> _previewTimestepFromPreviousMeasure;
    std::optional<std::chrono::steady_clock::time_point> _timepointFromPreviousMeasure;
    uint64_t _currentTimestep = 0;
    std::optional<int> _tpsFromPreviousMeasure;
};
