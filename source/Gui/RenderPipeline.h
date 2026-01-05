#pragma once

#include <EngineInterface/OpenGlGeometryBuffers.h>
#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "RenderStep.h"

// Contains RenderSteps that must be executed in order
struct RenderSequence
{
    MEMBER(RenderSequence, std::vector<RenderStep>, steps, {});

    RenderSequence& repetitions(int value)
    {
        _repetitions = value;
        return *this;
    }
    using RepetitionFunc = std::function<int(void)>;
    RenderSequence& repetitions(RepetitionFunc const& value)
    {
        _repetitions = value;
        return *this;
    }
    int getRepetitions() const
    {
        if (std::holds_alternative<int>(_repetitions)) {
            return std::get<int>(_repetitions);
        } else {
            return std::get<RepetitionFunc>(_repetitions)();
        }
    }
    std::variant<int, RepetitionFunc> _repetitions = 1;
};

// Contains RenderSequences that are independent
using RenderBlock = std::vector<RenderSequence>;

// Contains RenderBlocks that must be executed in order
using RenderBlocks = std::vector<RenderBlock>;

class _RenderPipeline
{
public:
    _RenderPipeline(RenderBlocks&& blocks);

    void resize(IntVector2D const& size);
    void execute();

private:
    void resizeTarget(TextureTarget const& target);

    void forEachStep(
        std::function<TextureTarget()> const& getTextureTarget,
        std::function<void(RenderStep& step, std::vector<unsigned int> const& textures, RenderTarget const& target)> const& executeStep);

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

    RenderBlocks _blocks;

    OpenGlGeometryBuffers _geometryBuffers;
    std::vector<TextureTarget> _textureTargets;
    std::optional<IntVector2D> _textureSize;
};
