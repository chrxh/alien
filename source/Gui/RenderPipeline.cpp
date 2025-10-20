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
        if (!step->getPreviousTargetSelection().has_value()) {
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
        auto vao = _geometryBuffers->getVaoForZones();
        auto vbo = _geometryBuffers->getVboForZones();

        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Setup vertex attributes for ZoneVertexData
        // Position (2 floats: x, y)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ZoneVertexData), (void*)0);
        glEnableVertexAttribArray(0);

        // Color (3 floats: r, g, b)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ZoneVertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // Shape type (1 int)
        glVertexAttribIPointer(2, 1, GL_INT, sizeof(ZoneVertexData), (void*)(5 * sizeof(float)));
        glEnableVertexAttribArray(2);

        // Dimension1 (1 float: radius or width)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(ZoneVertexData), (void*)(5 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(3);

        // Dimension2 (1 float: unused or height)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, sizeof(ZoneVertexData), (void*)(6 * sizeof(float) + sizeof(int)));
        glEnableVertexAttribArray(4);
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
        [](RenderStep&, std::vector<unsigned int> const&, RenderTarget const&, float) {});
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    std::set<TextureTarget> textureTargets;
    for (auto const& block : _blocks) {
        for (auto const& sequence : block) {
            for (auto const& step : sequence._steps) {
                if (step->getTextureTarget() != nullptr) {
                    step->resize(size);
                }
            }
        }
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

    forEachStep(
        [](RenderStep& step) { return step->getTextureTarget(); },
        [this,
         &generalRenderInfo](RenderStep& step, std::vector<unsigned int> const& textures, RenderTarget const& target, float scale) {
            step->execute(ExecutionParameters()
                              .geometryBuffers(_geometryBuffers)
                              .textures(textures)
                              .target(target)
                              .scale(scale) 
                              .renderInfo(generalRenderInfo)
                              .simulationFacade(_simulationFacade));
        });

    glBindFramebuffer(GL_FRAMEBUFFER, generalRenderInfo.screenFbo);
}

void _RenderPipeline::forEachStep(
    std::function<TextureTarget(RenderStep& step)> const& getTextureTarget,
    std::function<void(RenderStep& step, std::vector<unsigned> const& textures, RenderTarget const& target, float scale)> const& executeStep)
{
    std::vector<RenderTarget> previousBlockTargets;
    std::vector<float> previousBlockScales;
    for (size_t i = 0; i < _blocks.size(); ++i) {
        auto& block = _blocks.at(i);
        auto isLastBlock = (i == _blocks.size() - 1);

        std::vector<RenderTarget> blockTargets;
        std::vector<float> blockScales;
        for (size_t j = 0; j < block.size(); ++j) {
            auto& sequence = block.at(j);

            std::vector<RenderTarget> previousTargets = previousBlockTargets;
            std::vector<float> previousScales = previousBlockScales;
            for (int k = 0; k < sequence._repetitions; ++k) {
                for (size_t l = 0; l < sequence._steps.size(); ++l) {
                    auto& step = sequence._steps.at(l);

                    // Determine target
                    RenderTarget target;
                    float scale;
                    if (auto previousTarget = step->getPreviousTargetSelection()) {
                        target = previousTargets.at(previousTarget.value());
                        scale = previousScales.at(previousTarget.value());
                    } else {
                        if (!sequence.subsequentStepsHaveTarget(l) && isLastBlock) {
                            target = RenderTarget(ScreenTarget());
                            scale = 1.0f;
                        } else {
                            target = getTextureTarget(step);
                            scale = step->getTextureScale();
                        }
                    }
                    // Execute render step
                    executeStep(step, getTextures(previousTargets), target, scale);

                    // Current output is input for next step
                    previousTargets = {target};
                    previousScales = {scale};
                }
            }
            CHECK(previousTargets.size() == 1);
            blockTargets.emplace_back(previousTargets.front());
            blockScales.emplace_back(previousScales.front());
        }
        previousBlockTargets = blockTargets;
        previousBlockScales = blockScales;
    }
}
