#include "RenderPipeline.h"

#include <ranges>

#include "EngineInterface/GeometryBuffers.h"

#include "RenderStep.h"
#include "RenderStep.h"
#include "Shader.h"

_RenderPipeline::_RenderPipeline(SimulationFacade const& simulationFacade, RenderBlocks&& blocks)
    : _simulationFacade(simulationFacade)
    , _geometryBuffers(_GeometryBuffers::create())
    , _blocks(std::move(blocks))
{
    {
        auto vao = _geometryBuffers->getVaoForPointsAndLines();
        auto vbo = _geometryBuffers->getVbo();
        auto ebo = _geometryBuffers->getEboForLines();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for VertexData (same as PointRenderStep)
        // Position (2 floats)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats) - used for line colors
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Z-position (1 float) - used for lighting calculations
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(5 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Bind EBO (will be filled by CUDA later)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    }
    {
        auto vao = _geometryBuffers->getVaoForTriangles();
        auto vbo = _geometryBuffers->getVbo();
        auto ebo = _geometryBuffers->getEboForTriangles();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for VertexData (same as PointRenderStep)
        // Position (2 floats)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats) - not used for lines but needed for compatibility
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Z-position (1 float) - used for lighting in triangle rendering
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(5 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Bind EBO (will be filled by CUDA later)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    }

    CHECK(!_blocks.empty());
    CHECK(_blocks.back()._sequences.size() == 1);

    for (auto& sequence : _blocks.front()) {
        sequence.front()->setClearBackground(true);
    }

    // Setup render targets for each step in the pipeline
    auto getPreviousTarget = [&](size_t i, size_t j, size_t k) {
        if (k > 0) {
            // Previous step in the same sequence
            auto& prevStep = _blocks.at(i).at(j).at(k - 1);
            return prevStep->getTarget().value();
        } else if (i > 0) {
            // Last step in the last sequence of the previous block
            auto& prevBlock = _blocks.at(i - 1);
            auto& prevQueue = prevBlock.back();
            auto& prevStep = prevQueue.back();
            return prevStep->getTarget().value();
        } else {
            return RenderTarget(ScreenTarget());
        }
    };
    auto furtherTargetsAvailable = [&](size_t i, size_t j, size_t k) {
        // Checking current sequence
        for (size_t stepIdx = k + 1; stepIdx < _blocks.at(i).at(j).size(); ++stepIdx) {
            if (!_blocks.at(i).at(j).at(stepIdx)->isUsePreviousOutput()) {
                return true;
            }
        }
        // Checking subsequent sequences in following blocks
        for (size_t blockIdx = i + 1; blockIdx < _blocks.size(); ++blockIdx) {
            for (size_t sequenceIdx = 0; sequenceIdx < _blocks.at(blockIdx).size(); ++sequenceIdx) {
                for (size_t stepIdx = 0; stepIdx < _blocks.at(blockIdx).at(sequenceIdx).size(); ++stepIdx) {
                    if (!_blocks.at(blockIdx).at(sequenceIdx).at(stepIdx)->isUsePreviousOutput()) {
                        return true;
                    }
                }
            }
        }
        return false;
    };
    for (size_t i = 0; i < _blocks.size(); ++i) {
        auto& block = _blocks.at(i);

        for (size_t j = 0; j < block._sequences.size(); ++j) {
            auto& sequence = block._sequences.at(j);

            for (size_t k = 0; k < sequence.size(); ++k) {
                auto& step = sequence.at(k);
                if (step->isUsePreviousOutput()) {
                    step->setTarget(getPreviousTarget(i, j, k));
                } else {
                    if (furtherTargetsAvailable(i, j, k)) {
                        step->setTarget(RenderTarget(_TextureTarget::create()));
                    } else {
                        step->setTarget(RenderTarget(ScreenTarget()));
                    }
                }
            }
        }
    }
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    // Collect texture targets which needs to be resized
    std::set<TextureTarget> textureTargets;
    for (auto const& block : _blocks) {
        for (auto const& sequence : block._sequences) {
            for (auto const& step : sequence) {
                if (std::holds_alternative<TextureTarget>(step->getTarget().value())) {
                    textureTargets.insert(std::get<TextureTarget>(step->getTarget().value()));
                }
            }
        }
    }

    for (auto& textureTarget : textureTargets) {
        if (textureTarget->initialized) {
            glDeleteFramebuffers(1, &textureTarget->fbo);
            glDeleteTextures(1, &textureTarget->texture);
        } 
        // Init output texture
        glGenTextures(1, &textureTarget->texture);
        glBindTexture(GL_TEXTURE_2D, textureTarget->texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);
        
        // Init framebuffer
        glGenFramebuffers(1, &textureTarget->fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, textureTarget->fbo);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, textureTarget->texture, 0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
}

void _RenderPipeline::execute()
{
    // Copy vertex buffer from Cuda to OpenGL
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(_geometryBuffers);

    GeneralRenderInfo generalRenderInfo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &generalRenderInfo.screenFbo);

    std::optional<RenderBlock> prevBlock;
    for (auto const& block : _blocks) {
        for (auto const& sequence : block._sequences) {
            std::vector<RenderStep> dependentSteps;
            if (prevBlock.has_value()) {
                for (auto const& dependentQueue : prevBlock.value()) {
                    dependentSteps.emplace_back(dependentQueue.back());
                }
            }
            sequence.front()->execute(_geometryBuffers, generalRenderInfo, _simulationFacade, dependentSteps);
            for (auto const& [prevStep, step] : std::views::zip(sequence, sequence | std::views::drop(1))) {
                step->execute(_geometryBuffers, generalRenderInfo, _simulationFacade, {prevStep});
            }
        }
        prevBlock = block;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, generalRenderInfo.screenFbo);
}
