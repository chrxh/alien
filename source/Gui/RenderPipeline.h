#pragma once

#include "EngineInterface/GeometryBuffers.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"

// Contains RenderSteps that must be executed in order
struct RenderSequence
{
    MEMBER(RenderSequence, std::vector<RenderStep>, steps, {});
    MEMBER(RenderSequence, int, repetitions, 1);

    bool subsequentStepsHaveTarget(size_t index) const;
};

// Contains RenderSequences that are independent
using RenderBlock = std::vector<RenderSequence>;

// Contains RenderBlocks that must be executed in order
using RenderBlocks = std::vector<RenderBlock>;

class _RenderPipeline
{
public:
    _RenderPipeline(SimulationFacade const& simulationFacade, RenderBlocks&& blocks);

    void resize(IntVector2D const& size);
    void execute();

private:
    SimulationFacade _simulationFacade;
    RenderBlocks _blocks;
    
    GeometryBuffers _geometryBuffers;
};

