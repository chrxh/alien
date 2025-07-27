#pragma once

#include <chrono>

#include "EngineInterface/Definitions.h"
#include "EngineInterface/Descriptions.h"
#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"

class _SimulatedPreviewWidget
{
public:
    static SimulatedPreviewWidget create(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData);

    void process();

private:
    _SimulatedPreviewWidget(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData);

    void initPreview();
    void calcPreview();
    void showPreview();

    SimulationFacade _simulationFacade;
    PreviewDescriptionWidget _previewWidget;

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    std::optional<GenomeDescription> _lastGenome;
    std::optional<uint64_t> _lastPreviewTimestep;
    std::optional<std::chrono::steady_clock::time_point> _lastTimepoint;
    std::optional<int> _lastTps;
};
