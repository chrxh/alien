#pragma once

#include "EngineInterface/RenderData.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"

class _RenderPipeline
{
public:
    _RenderPipeline(SimulationFacade const& simulationFacade);

    void addStep(RenderStep2 const& step);

    void resize(IntVector2D const& size);
    void execute();

private:
    RenderStep2 findNextStep(std::set<RenderStep2> const& finishedSteps) const;

    SimulationFacade _simulationFacade;
    std::vector<RenderStep2> _steps;

    NumRenderObjects _numObjects;
};
