#pragma once

#include "EngineInterface/GeometryBuffers.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"

// Contains RenderSteps that must be executed in order
using RenderSequence = std::vector<RenderStep>;

// Contains RenderSequences that are independent
struct RenderBlock
{
    MEMBER(RenderBlock, std::vector<RenderSequence>, sequences, {});
    MEMBER(RenderBlock, int, repetitions, 1);
};

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

