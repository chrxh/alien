#include "RenderPipeline.h"

#include <ranges>

#include <boost/range/adaptor/indexed.hpp>

#include <EngineInterface/GeometryBuffers.h>
#include <EngineInterface/SimulationFacade.h>

#include "RenderStep.h"
#include "Shader.h"
#include "Viewport.h"

namespace
{
    auto constexpr IntermediateFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    auto constexpr ScreenFormat = VK_FORMAT_R8G8B8A8_UNORM;
}

_RenderPipeline::_RenderPipeline(RenderBlocks&& blocks)
    : _geometryBuffers(_VulkanGeometryBuffers::create())
    , _blocks(std::move(blocks))
    , _screenTarget(_TextureTarget::create())
{
    // Check for supported pipeline structure
    CHECK(!_blocks.empty());
    CHECK(_blocks.back().size() == 1);
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    _textureSize = size;
    for (auto const& textureTarget : _textureTargets) {
        resizeTarget(textureTarget);
    }
    _screenTarget->resize(size, ScreenFormat);
}

namespace
{
    std::vector<TextureTarget> getTextures(std::vector<RenderTarget> const& targets)
    {
        std::vector<TextureTarget> result;
        for (auto const& target : targets) {
            if (std::holds_alternative<TextureTarget>(target)) {
                result.emplace_back(std::get<TextureTarget>(target));
            }
        }
        return result;
    }
}

VulkanImage& _RenderPipeline::execute(VkCommandBuffer commandBuffer, std::optional<TextureTarget> const& finalTarget)
{
    _finalTarget = finalTarget ? RenderTarget(*finalTarget) : RenderTarget(ScreenTarget());

    // Copy vertex buffer from Cuda to Vulkan
    _SimulationFacade::get()->tryCopyBuffersFromCudaToRenderer(_geometryBuffers, Viewport::get().getVisibleWorldRect());
    _geometryBuffers->prepareForRendering(commandBuffer);

    GeneralRenderInfo generalRenderInfo{.commandBuffer = commandBuffer};

    auto simParameters = std::make_shared<SimulationParameters>(_SimulationFacade::get()->getSimulationParameters());
    int currentTextureTargetIndex = 0;
    forEachStep(
        [this, &currentTextureTargetIndex] {
            if (currentTextureTargetIndex < _textureTargets.size()) {
                return _textureTargets.at(currentTextureTargetIndex++);
            } else {
                auto result = _TextureTarget::create();
                resizeTarget(result);
                _textureTargets.emplace_back(result);
                ++currentTextureTargetIndex;
                return result;
            }
        },
        [this, &generalRenderInfo, &simParameters](RenderStep& step, std::vector<TextureTarget> const& textures, RenderTarget const& target) {
            // Merge inputTextures from step parameters with textures from previous targets
            auto allTextures = step->getInputTextures();
            allTextures.insert(allTextures.end(), textures.begin(), textures.end());

            auto textureTarget = std::holds_alternative<ScreenTarget>(target) ? _screenTarget : std::get<TextureTarget>(target);
            step->execute(ExecutionParameters()
                              .geometryBuffers(_geometryBuffers)
                              .textures(allTextures)
                              .target(textureTarget)
                              .renderInfo(generalRenderInfo)
                              .simulationFacade(_SimulationFacade::get())
                              .simulationParameters(simParameters));
        });

    return std::holds_alternative<ScreenTarget>(_finalTarget) ? _screenTarget->color : std::get<TextureTarget>(_finalTarget)->color;
}

void _RenderPipeline::resizeTarget(TextureTarget const& target)
{
    CHECK(_textureSize.has_value());

    target->resize(*_textureSize, IntermediateFormat);
}

void _RenderPipeline::forEachStep(
    std::function<TextureTarget()> const& getTextureTarget,
    std::function<void(RenderStep&, std::vector<TextureTarget> const&, RenderTarget const&)> const& executeStep)
{
    std::map<RenderTarget, TargetInfo> usedTargets;

    std::vector<RenderTarget> previousBlockTargets;
    for (size_t i = 0; i < _blocks.size(); ++i) {
        auto& block = _blocks.at(i);
        auto isLastBlock = (i == _blocks.size() - 1);

        std::vector<RenderTarget> blockTargets;
        for (size_t j = 0; j < block.size(); ++j) {
            auto& sequence = block.at(j);

            std::vector<RenderTarget> previousTargets = previousBlockTargets;
            auto repetitions = sequence.getRepetitions();
            for (int k = 0; k < repetitions; ++k) {
                for (size_t l = 0; l < sequence._steps.size(); ++l) {
                    auto& step = sequence._steps.at(l);

                    // Determine target
                    auto target = determineRenderTarget(step, sequence, block, i, j, k, l, isLastBlock, getTextureTarget, previousTargets, usedTargets);

                    // Execute render step
                    executeStep(step, getTextures(previousTargets), target);

                    // Current output is input for next step
                    previousTargets = {target};
                }
            }
            CHECK(previousTargets.size() == 1);
            blockTargets.emplace_back(previousTargets.front());
        }
        previousBlockTargets = blockTargets;
    }
}

namespace
{
    bool subsequentStepsHaveTarget(RenderSequence const& sequence, size_t stepIndex)
    {
        for (size_t i = stepIndex + 1; i < sequence._steps.size(); ++i) {
            auto const& step = sequence._steps.at(i);
            if (!step->getPreviousTargetSelection().has_value()) {
                return true;
            }
        }
        return false;
    }

}

RenderTarget _RenderPipeline::determineRenderTarget(
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
    std::map<RenderTarget, TargetInfo>& usedTargets)
{
    RenderTarget target;
    if (auto previousTarget = step->getPreviousTargetSelection()) {
        target = previousTargets.at(previousTarget.value());
        if (std::holds_alternative<TextureTarget>(target) && usedTargets.contains(target)) {
            TargetInfo targetInfo{
                .block = blockIndex,
                .lastStepInSequence = (stepIndex == sequence._steps.size() - 1) && (repetitionIndex == sequence.getRepetitions() - 1),
            };
            usedTargets.at(target) = targetInfo;
        }
    } else {
        if (!subsequentStepsHaveTarget(sequence, stepIndex) && isLastBlock) {
            target = _finalTarget;
        } else {

            auto reuseTarget = false;
            for (auto const& [usedTarget, targetInfo] : usedTargets) {

                // Do not reuse targets which are used as input
                auto findResult = std::ranges::find(previousTargets, usedTarget);
                if (findResult != previousTargets.end()) {
                    continue;
                }
                // Do not reuse targets from the same block since they could be used in the next block
                if (targetInfo.block == blockIndex && targetInfo.lastStepInSequence && !isLastBlock) {
                    continue;
                }
                // Do not reuse targets from the previous block if they are still used in the current block
                if (targetInfo.block == blockIndex - 1 && targetInfo.lastStepInSequence
                    && (stepIndex < sequence._steps.size() - 1 || repetitionIndex < sequence.getRepetitions() - 1)) {
                    continue;
                }
                if (targetInfo.block == blockIndex - 1 && targetInfo.lastStepInSequence
                    && (sequenceIndex < block.size() - 1 || repetitionIndex < sequence.getRepetitions() - 1)) {
                    continue;
                }
                target = usedTarget;
                reuseTarget = true;
            }

            if (!reuseTarget || std::ranges::find(previousTargets, target) != previousTargets.end()) {
                target = getTextureTarget();
            }

            TargetInfo targetInfo{
                .block = blockIndex,
                .lastStepInSequence = (stepIndex == sequence._steps.size() - 1) && (repetitionIndex == sequence.getRepetitions() - 1),
            };
            usedTargets.insert_or_assign(target, targetInfo);
        }
    }
    return target;
}
