#pragma once

#include "EngineInterface/RenderBuffers.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"

class _RenderPipeline
{
public:
    _RenderPipeline(SimulationFacade const& simulationFacade);

    void addStep(RenderStep const& step);

    void resize(IntVector2D const& size);
    void execute();

private:
    RenderStep findNextStep(std::set<RenderStep> const& finishedSteps) const;

    SimulationFacade _simulationFacade;
    std::vector<RenderStep> _steps;

    NumRenderObjects _numObjects;

    GeometrySource _geometrySource;
    RenderBuffers _renderBuffers;
};
