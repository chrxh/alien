#include "RenderPipeline.h"

#include <ranges>

#include <boost/range/adaptor/indexed.hpp>

#include <EngineInterface/GeometryBuffers.h>

#include "RenderStep.h"
#include "Shader.h"
#include "Viewport.h"
#include <EngineInterface/SimulationFacade.h>

_RenderPipeline::_RenderPipeline(RenderBlocks&& blocks)
    : _geometryBuffers(_GeometryBuffers::create())
    , _blocks(std::move(blocks))
{
    {
        auto vao = _geometryBuffers->getVaoForPointsAndLines();
        auto vbo = _geometryBuffers->getVboForCells();
        auto ebo = _geometryBuffers->getEboForLines();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for CellVertexData (same as PointRenderStep)
        // Position (3 floats: x, y, z)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CellVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats: r, g, b)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CellVertexData), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // States (1 int)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(CellVertexData), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Bind EBO (will be filled by CUDA later)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    }
    {
        auto vao = _geometryBuffers->getVaoForTriangles();
        auto vbo = _geometryBuffers->getVboForCells();
        auto ebo = _geometryBuffers->getEboForTriangles();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for CellVertexData (same as PointRenderStep)
        // Position (3 floats: x, y, z)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(CellVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats: r, g, b)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(CellVertexData), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // States (1 int)
        glVertexAttribIPointer(2, 1, GL_INT, sizeof(CellVertexData), (void*)(6 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Bind EBO (will be filled by CUDA later)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    }
    {
        auto vao = _geometryBuffers->getVaoForEnergyParticles();
        auto vbo = _geometryBuffers->getVboForEnergyParticles();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for EnergyParticleVertexData
        // Position (3 floats: x, y, z)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(EnergyParticleVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats: r, g, b)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(EnergyParticleVertexData), (void*)(3 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    {
        auto vao = _geometryBuffers->getVaoForLocations();
        auto vbo = _geometryBuffers->getVboForLocations();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for LocationVertexData
        // Position (2 floats: x, y)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(LocationVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats: r, g, b)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(LocationVertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Shape type (1 int)
        glVertexAttribIPointer(2, 1, GL_INT, sizeof(LocationVertexData), (void*)(5 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Dimension1 (1 float: radius or width)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(LocationVertexData), (void*)(5 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(3);

        // Dimension2 (1 float: unused or height)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(LocationVertexData), (void*)(6 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(4);

        // FadeoutRadius (1 float)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(LocationVertexData), (void*)(7 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(5);

        // Opacity (1 float)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, sizeof(LocationVertexData), (void*)(8 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(6);
    }
    {
        auto vao = _geometryBuffers->getVaoForSelectedObjects();
        auto vbo = _geometryBuffers->getVboForSelectedObjects();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for SelectedObjectVertexData
        // Position (2 floats: x, y)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(SelectedObjectVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // HasSignalRestriction (1 int)
        glVertexAttribIPointer(1, 1, GL_INT, sizeof(SelectedObjectVertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // StartAngle (1 float)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(SelectedObjectVertexData), (void*)(2 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(2);

        // EndAngle (1 float)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(SelectedObjectVertexData), (void*)(3 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(3);
    }
    {
        auto vao = _geometryBuffers->getVaoForSelectedConnections();
        auto vbo = _geometryBuffers->getVboForSelectedConnections();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for ConnectionArrowVertexData
        // Position (2 floats: x, y)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ConnectionArrowVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats: r, g, b)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ConnectionArrowVertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Arrow flags (1 int)
        glVertexAttribIPointer(2, 1, GL_INT, sizeof(ConnectionArrowVertexData), (void*)(5 * sizeof(float)));
        glEnableVertexAttribArray(2);
    }
    {
        auto vao = _geometryBuffers->getVaoForAttackEvents();
        auto vbo = _geometryBuffers->getVboForAttackEvents();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for AttackEventVertexData
        // Position (2 floats: x, y)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(AttackEventVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats: r, g, b)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(AttackEventVertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }
    {
        auto vao = _geometryBuffers->getVaoForDetonationEvents();
        auto vbo = _geometryBuffers->getVboForDetonationEvents();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for DetonationEventVertexData
        // Position (2 floats: x, y)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(DetonationEventVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Radius (1 float)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(DetonationEventVertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    // Check for supported pipeline structure
    CHECK(!_blocks.empty());
    CHECK(_blocks.back().size() == 1);

    //// Create texture targets for all steps which need one
    //forEachStep(
    //    [this]() {
    //        auto result = _TextureTarget::create();
    //        _textureTargets.emplace_back(result);
    //        return result;
    //    },
    //    [](RenderStep&, std::vector<unsigned int> const&, RenderTarget const&) {});
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    _textureSize = size;
    for (auto const& textureTarget : _textureTargets) {
        resizeTarget(textureTarget);
    }
}

namespace
{
    std::vector<unsigned int> getTextures(std::vector<RenderTarget> const& targets)
    {
        std::vector<unsigned int> result;
        for (auto const& target : targets) {
            if (std::holds_alternative<TextureTarget>(target)) {
                result.emplace_back(std::get<TextureTarget>(target)->texture);
            }
        }
        return result;
    }
}

void _RenderPipeline::execute()
{
    // Copy vertex buffer from Cuda to OpenGL
    _SimulationFacade::get()->tryCopyBuffersFromCudaToOpenGL(_geometryBuffers, Viewport::get().getVisibleWorldRect());

    GeneralRenderInfo generalRenderInfo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &generalRenderInfo.screenFbo);

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
        [this, &generalRenderInfo, &simParameters](RenderStep& step, std::vector<unsigned int> const& textures, RenderTarget const& target) {
            // Merge inputTextures from step parameters with textures from previous targets
            auto allTextures = step->getInputTextures();
            allTextures.insert(allTextures.end(), textures.begin(), textures.end());

            step->execute(ExecutionParameters()
                              .geometryBuffers(_geometryBuffers)
                              .textures(allTextures)
                              .target(target)
                              .renderInfo(generalRenderInfo)
                              .simulationFacade(_SimulationFacade::get())
                              .simulationParameters(simParameters));
        });

    glBindFramebuffer(GL_FRAMEBUFFER, generalRenderInfo.screenFbo);
}

void _RenderPipeline::resizeTarget(TextureTarget const& target)
{
    CHECK(_textureSize.has_value());

    if (target->initialized) {
        glDeleteFramebuffers(1, &target->fbo);
        glDeleteTextures(1, &target->texture);
    }
    // Init output texture
    glGenTextures(1, &target->texture);
    glBindTexture(GL_TEXTURE_2D, target->texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, _textureSize->x, _textureSize->y, 0, GL_RGBA, GL_FLOAT, NULL);

    // Init framebuffer
    glGenFramebuffers(1, &target->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, target->fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, target->texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void _RenderPipeline::forEachStep(
    std::function<TextureTarget()> const& getTextureTarget,
    std::function<void(RenderStep&, std::vector<unsigned> const&, RenderTarget const&)> const& executeStep)
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
            target = RenderTarget(ScreenTarget());
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
