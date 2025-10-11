#include "RenderPipeline.h"

#include "RenderStep.h"
#include "Shader.h"

_RenderPipeline::_RenderPipeline(SimulationFacade const& simulationFacade)
    : _simulationFacade(simulationFacade)
{}

void _RenderPipeline::addStep(RenderStep const& step)
{
    _steps.emplace_back(step);
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    for (auto const& step : _steps) {
        step->resize(size);
    }
}

void _RenderPipeline::execute()
{
    // Start with point renderer
    RenderStep pointRenderStep;
    for (auto const& step : _steps) {
        if (std::dynamic_pointer_cast<_PointRenderStep>(step)) {
            pointRenderStep = step;
            break;
        }
    }
    CHECK(pointRenderStep);
    auto vbo = pointRenderStep->getShader()->getVbo();

    // Find line renderer to get EBO
    RenderStep lineRenderStep;
    for (auto const& step : _steps) {
        if (auto lineStep = std::dynamic_pointer_cast<_LineRenderStep>(step)) {
            lineRenderStep = step;
            break;
        }
    }
    CHECK(lineRenderStep);
    auto ebo = lineRenderStep->getShader()->getEbo();

    // Copy vertex buffer from Cuda to OpenGL
    RenderBuffers renderBuffers{.vboForPoints = vbo, .eboForLines = ebo};
    auto numRenderObjects = _simulationFacade->tryCopyBuffersFromCudaToOpenGL(renderBuffers);
    if (numRenderObjects.has_value()) {
        _numObjects = *numRenderObjects;
    }

    GLint screenFbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &screenFbo);

    std::set<RenderStep> finishedSteps;
    do {
        finishedSteps.insert(pointRenderStep);
        auto nextRenderStep = findNextStep(finishedSteps);
        auto isFinalStep = nextRenderStep == nullptr;
        RenderTarget target = isFinalStep ? RenderTarget(ScreenTarget{.fbo = screenFbo}) : RenderTarget(TextureTarget{});
        pointRenderStep->execute(target, _numObjects, _simulationFacade);
        pointRenderStep = nextRenderStep;
    } while (pointRenderStep);
}

namespace
{
    // Proofs if steps1 is contained in steps2
    bool isContained(std::vector<RenderStep> const& steps1, std::set<RenderStep> const& steps2)
    {
        for (auto const& step : steps1) {
            if (!steps2.contains(step)) {
                return false;
            }
        }
        return true;
    }
}

RenderStep _RenderPipeline::findNextStep(std::set<RenderStep> const& finishedSteps) const
{
    std::vector<RenderStep> result;
    for (auto const& candidate : _steps) {
        if (finishedSteps.contains(candidate)) {
            continue;
        }
        auto const& prevSteps = candidate->getDependentSteps();
        if (isContained(prevSteps, finishedSteps)) {
            return candidate;
        }
    }
    return nullptr;
}
