#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/GenomeDescription.h"

#include "Definitions.h"

class _SimulatedPreviewWidget
{
public:
    static SimulatedPreviewWidget create(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData);

    void process();

private:
    _SimulatedPreviewWidget(SimulationFacade const& simulationFacade, GenomeTabEditData const& editData);

    SimulationFacade _simulationFacade;

    GenomeTabEditData _editData;
    GenomeTabLayoutData _layoutData;

    std::optional<GenomeDescription> _lastGenome;
    PreviewDescriptionWidget _previewWidget;
};
