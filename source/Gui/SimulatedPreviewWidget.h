#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"

class _SimulatedPreviewWidget
{
public:
    static SimulatedPreviewWidget
    create(SimulationFacade const& simulationFacade, GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData);

    void process();

private:
    _SimulatedPreviewWidget(SimulationFacade const& simulationFacade, GenomeWindowEditData const& genomeEditData, GenomeTabEditData const& editData);

    void initPreview();
    void continuePreview();
    void calcPreview();
    void showPreview();

    SimulationFacade _simulationFacade;
    PreviewDescriptionWidget _previewWidget;

    GenomeWindowEditData _genomeEditData;
    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    CollectionDescription _previewData;
    std::optional<GenomeDescription> _genomeFromPreviousFrame;
    std::optional<uint64_t> _previewTimestepFromPreviousMeasure;
    std::optional<std::chrono::steady_clock::time_point> _timepointFromPreviousMeasure;
    std::optional<int> _tpsFromPreviousMeasure;
};
