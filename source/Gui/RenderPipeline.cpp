#include "RenderPipeline.h"

#include "RenderStep.h"
#include "RenderStep2.h"
#include "Shader.h"

_RenderPipeline::_RenderPipeline(SimulationFacade const& simulationFacade)
    : _simulationFacade(simulationFacade)
{}

void _RenderPipeline::addStep(RenderStep2 const& step)
{
    _steps.emplace_back(step);
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    std::set<TextureTarget> textureTargets;

    for (auto const& step : _steps) {
        if (std::holds_alternative<TextureTarget>(step->getTarget())) {
            textureTargets.insert(std::get<TextureTarget>(step->getTarget()));
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
    // Start with point renderer
    RenderStep2 startRenderStep;
    for (auto const& step : _steps) {
        if (std::dynamic_pointer_cast<_PointRenderStep2>(step)) {
            startRenderStep = step;
            break;
        }
    }
    CHECK(startRenderStep);
    auto vbo = std::get<GeometrySource>(startRenderStep->getSource()).vbo;

    //// Find line renderer to get EBO
    //RenderStep lineRenderStep;
    //for (auto const& step : _steps) {
    //    if (auto lineStep = std::dynamic_pointer_cast<_LineRenderStep>(step)) {
    //        lineRenderStep = step;
    //        break;
    //    }
    //}
    //CHECK(lineRenderStep);
    //auto vao = lineRenderStep->getShader()->getVao();
    //auto ebo = lineRenderStep->getShader()->getEbo();

    // Copy vertex buffer from Cuda to OpenGL
    RenderBuffers renderBuffers{.vboForPoints = vbo/*, .vaoForLines = vao, .eboForLines = ebo*/};
    auto numRenderObjects = _simulationFacade->tryCopyBuffersFromCudaToOpenGL(renderBuffers);
    if (numRenderObjects.has_value()) {
        _numObjects = *numRenderObjects;
    }

    GeneralRenderInfo generalRenderInfo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &generalRenderInfo.screenFbo);

    auto currentRenderStep = startRenderStep;
    std::set<RenderStep2> finishedSteps;
    do {
        finishedSteps.insert(currentRenderStep);
        auto nextRenderStep = findNextStep(finishedSteps);
        if (auto pointRenderStep = std::dynamic_pointer_cast<_PointRenderStep2>(currentRenderStep)) {
            pointRenderStep->execute(_numObjects.vertices, generalRenderInfo, _simulationFacade);
        }
        currentRenderStep = nextRenderStep;
    } while (currentRenderStep);
}

namespace
{
    // Proofs if steps1 is contained in steps2
    bool isContained(std::vector<RenderStep2> const& steps1, std::set<RenderStep2> const& steps2)
    {
        for (auto const& step : steps1) {
            if (!steps2.contains(step)) {
                return false;
            }
        }
        return true;
    }
}

RenderStep2 _RenderPipeline::findNextStep(std::set<RenderStep2> const& finishedSteps) const
{
    std::vector<RenderStep2> result;
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
