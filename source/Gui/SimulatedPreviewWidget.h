#pragma once

#include "EngineInterface/Definitions.h"
#include "EngineInterface/GenomeDescriptions.h"

#include "Definitions.h"

class _SimulatedPreviewWidget
{
public:
    static SimulatedPreviewWidget create(SimulationFacade const& simulationFacade, CreatureTabEditData const& editData);

    void process();

private:
    _SimulatedPreviewWidget(SimulationFacade const& simulationFacade, CreatureTabEditData const& editData);

    SimulationFacade _simulationFacade;

    CreatureTabEditData _editData;
    CreatureTabLayoutData _layoutData;

    std::optional<GenomeDescription_New> _lastGenome;
};
