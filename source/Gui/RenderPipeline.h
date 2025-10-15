#pragma once

#include "EngineInterface/GeometryBuffers.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"

using RenderQueue = std::vector<RenderStep>;
using RenderBlock = std::vector<RenderQueue>;
using RenderBlocks = std::vector<RenderBlock>;

class _RenderPipeline
{
public:
    _RenderPipeline(SimulationFacade const& simulationFacade, RenderBlocks&& blocks);

    void resize(IntVector2D const& size);
    void execute();

private:
    void executeStep(RenderStep const& step, GeneralRenderInfo const& generalRenderInfo, std::vector<RenderStep> const& dependentSteps);

    SimulationFacade _simulationFacade;
    RenderBlocks _blocks;
    
    GeometryBuffers _geometryBuffers;
};

