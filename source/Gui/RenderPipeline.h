#pragma once

#include "EngineInterface/GeometryBuffers.h"
#include "EngineInterface/SimulationFacade.h"

#include "Definitions.h"
#include "RenderStep.h"

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
    void forEachStep(
        std::function<TextureTarget()> const& getTextureTarget,
        std::function<void(RenderStep& step, std::vector<unsigned int> const& textures, RenderTarget const& target)> const&
            executeStep);

    struct TargetInfo
    {
        size_t block = 0;
        bool lastStepInSequence = false;
    };
    RenderTarget determineRenderTarget(
        RenderStep const& step,
        RenderSequence const& sequence,
        RenderBlock const& block,
        size_t blockIndex,
        size_t sequenceIndex,
        size_t repetitionIndex,
        size_t stepIndex,
        bool isLastBlock,
        std::function<TextureTarget()> const& getTextureTarget,
        std::vector<RenderTarget> const& previousTargets,
        std::map<RenderTarget, TargetInfo>& usedTargets);

    SimulationFacade _simulationFacade;
    RenderBlocks _blocks;
    
    GeometryBuffers _geometryBuffers;
    std::vector<TextureTarget> _textureTargets;
};

