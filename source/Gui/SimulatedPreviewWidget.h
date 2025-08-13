#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"

class _SimulatedPreviewWidget
{
public:
    static SimulatedPreviewWidget create(
        SimulationFacade const& simulationFacade,
        PreviewDescriptionSettings const& previewSettings,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void process();

private:
    _SimulatedPreviewWidget(
        SimulationFacade const& simulationFacade,
        PreviewDescriptionSettings const& settings,
        GenomeWindowEditData const& genomeEditData,
        GenomeTabEditData const& editData);

    void createSubGenomesForPreview();
    void setPreviewData();
    void calcPreview();
    void showPreview();

    SimulationFacade _simulationFacade;
    PreviewDescriptionWidget _previewWidget;

    PreviewDescriptionSettings _settings;
    GenomeWindowEditData _genomeEditData;
    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    std::vector<GenomeDescriptionWithStartGeneIndex> _subGenomesForPreview;

    GenomeDescription _genomeForPreview;
    int _rootGeneIndex = 0;

    std::optional<GenomeDescription> _genomeFromPreviousFrame;
    std::optional<int> _selectedGeneIndexFromPreviousFrame;

    std::optional<uint64_t> _previewTimestepFromPreviousMeasure;
    std::optional<std::chrono::steady_clock::time_point> _timepointFromPreviousMeasure;
    std::optional<int> _tpsFromPreviousMeasure;
};
