#include "RenderPipeline.h"

#include <ranges>

#include <boost/range/adaptor/indexed.hpp>

#include "EngineInterface/GeometryBuffers.h"

#include "RenderStep.h"
#include "Shader.h"

bool RenderSequence::subsequentStepsHaveTarget(size_t index) const
{
    for (size_t i = index + 1; i < _steps.size(); ++i) {
        auto const& step = _steps.at(i);
        if (!step->isUsePreviousTarget()) {
            return true;
        }
    }
    return false;
}

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
    CHECK(_blocks.back().size() == 1);

    // Create texture targets for all steps which need one
    forEachStep(
        [](RenderStep& step) {
            auto result = _TextureTarget::create();
            step->setTextureTarget(result);
            return result;
        },
        [](RenderStep&, std::vector<unsigned int> const&, bool, RenderTarget const&) {});
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    // Collect texture targets which needs to be resized
    std::set<TextureTarget> textureTargets;
    for (auto const& block : _blocks) {
        for (auto const& sequence : block) {
            for (auto const& step : sequence._steps) {
                if (step->getTextureTarget().has_value()) {
                    textureTargets.insert(step->getTextureTarget().value());
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
    _simulationFacade->tryCopyBuffersFromCudaToOpenGL(_geometryBuffers);

    GeneralRenderInfo generalRenderInfo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &generalRenderInfo.screenFbo);

    forEachStep([](RenderStep& step) {
            return step->getTextureTarget().value();
        },
        [this, &generalRenderInfo](RenderStep& step, std::vector<unsigned int> const& textures, bool clearBackground, RenderTarget const& target) {
            step->execute(_geometryBuffers, textures, clearBackground, target, generalRenderInfo, _simulationFacade);
        });

    glBindFramebuffer(GL_FRAMEBUFFER, generalRenderInfo.screenFbo);
}

void _RenderPipeline::forEachStep(
    std::function<TextureTarget(RenderStep& step)> const& getTextureTarget,
    std::function<void(RenderStep& step, std::vector<unsigned> const& textures, bool clearBackground, RenderTarget const& target)> const& executeStep)
{
    std::vector<RenderTarget> previousBlockTargets;
    for (size_t i = 0; i < _blocks.size(); ++i) {
        auto& block = _blocks.at(i);
        auto isLastBlock = (i == _blocks.size() - 1);

        std::vector<RenderTarget> blockTargets;
        for (size_t j = 0; j < block.size(); ++j) {
            auto& sequence = block.at(j);

            std::vector<RenderTarget> previousTargets = previousBlockTargets;
            for (int k = 0; k < sequence._repetitions; ++k) {
                for (size_t l = 0; l < sequence._steps.size(); ++l) {
                    auto& step = sequence._steps.at(l);

                    // Determine target
                    RenderTarget target;
                    if (!step->isUsePreviousTarget()) {
                        if (!sequence.subsequentStepsHaveTarget(l) && isLastBlock) {
                            target = RenderTarget(ScreenTarget());
                        } else {
                            target = getTextureTarget(step);
                        }
                    } else {
                        CHECK(previousTargets.size() == 1);
                        target = previousTargets.front();
                    }
                    // Execute render step
                    auto clearBackground = i == 0 && k == 0 && l == 0;
                    executeStep(step, getTextures(previousTargets), clearBackground, target);

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
