#pragma once

#include <EngineInterface/GeometryBuffers.h>
#include <EngineInterface/SimulationFacade.h>

#include "Definitions.h"
#include "RenderStep.h"
#include "VulkanGeometryBuffers.h"

// Contains RenderSteps that must be executed in order
struct RenderSequence
{
    MEMBER(RenderSequence, std::vector<RenderStep>, steps, {});

    RenderSequence& repetitions(int value)
    {
        _repetitions = value;
        return *this;
    }
    using RepetitionFunc = std::function<int(RenderView const&)>;
    RenderSequence& repetitions(RepetitionFunc const& value)
    {
        _repetitions = value;
        return *this;
    }
    int getRepetitions(RenderView const& view) const
    {
        if (std::holds_alternative<int>(_repetitions)) {
            return std::get<int>(_repetitions);
        } else {
            return std::get<RepetitionFunc>(_repetitions)(view);
        }
    }
    std::variant<int, RepetitionFunc> _repetitions = 1;
};

// Contains RenderSequences that are independent
using RenderBlock = std::vector<RenderSequence>;

// Contains RenderBlocks that must be executed in order
using RenderBlocks = std::vector<RenderBlock>;

class _RenderGraph
{
public:
    _RenderGraph(RenderBlocks&& blocks);

    // Copies the visible simulation data into the geometry buffers, which the GPU must no longer use
    void updateGeometry(RealRect const& visibleWorldRect);

    // Records the rendering of the simulation into the command buffer and returns the image of the final target.
    // Without a final target, the graph renders into an own image of the view size.
    // The images take a new view size only here, so that resizing the window does not create images for every intermediate size.
    VulkanImage& execute(VkCommandBuffer commandBuffer, RenderView const& view, std::optional<TextureTarget> const& finalTarget = std::nullopt);

private:
    void applyViewSize(bool withScreenTarget);
    void resizeTarget(TextureTarget const& target);

    void forEachStep(
        std::function<TextureTarget()> const& getTextureTarget,
        std::function<void(RenderStep& step, std::vector<TextureTarget> const& textures, RenderTarget const& target)> const& executeStep);

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

    VulkanGeometryBuffers _geometryBuffers;
    RenderBlocks _blocks;

    RenderView _view;
    RenderTarget _finalTarget = ScreenTarget();

    TextureTarget _screenTarget;
    std::vector<TextureTarget> _textureTargets;
    std::optional<IntVector2D> _textureSize;
};
