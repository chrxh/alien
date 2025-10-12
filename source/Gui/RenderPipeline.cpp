#include "RenderPipeline.h"

#include "EngineInterface/GeometryBuffers.h"

#include "RenderStep.h"
#include "RenderStep2.h"
#include "Shader.h"

_RenderPipeline::_RenderPipeline(SimulationFacade const& simulationFacade)
    : _simulationFacade(simulationFacade)
    , _geometryBuffers(_GeometryBuffers::create())
{
    auto vao = _geometryBuffers->getVao();
    auto vbo = _geometryBuffers->getVbo();
    auto ebo = _geometryBuffers->getEbo();
    
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // Setup vertex attributes for VertexData (same as PointRenderStep)
    // Position (2 floats)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Color (3 floats) - not used for lines but needed for compatibility
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    // Bind EBO (will be filled by CUDA later)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
}

void _RenderPipeline::addStep(RenderStep const& step)
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
    RenderStep startRenderStep;
    for (auto const& step : _steps) {
        if (std::dynamic_pointer_cast<_PointRenderStep>(step)) {
            startRenderStep = step;
            break;
        }
    }
    CHECK(startRenderStep);

    // Copy vertex buffer from Cuda to OpenGL
    auto numRenderObjects = _simulationFacade->tryCopyBuffersFromCudaToOpenGL(_geometryBuffers);
    if (numRenderObjects.has_value()) {
        _numObjects = *numRenderObjects;
    }

    GeneralRenderInfo generalRenderInfo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &generalRenderInfo.screenFbo);

    auto currentRenderStep = startRenderStep;
    std::set<RenderStep> finishedSteps;
    do {
        finishedSteps.insert(currentRenderStep);
        auto nextRenderStep = findNextStep(finishedSteps);
        if (auto pointRenderStep = std::dynamic_pointer_cast<_PointRenderStep>(currentRenderStep)) {
            pointRenderStep->execute(_numObjects.vertices, _geometryBuffers, generalRenderInfo, _simulationFacade);
        }
        if (auto lineRenderStep = std::dynamic_pointer_cast<_LineRenderStep>(currentRenderStep)) {
            lineRenderStep->execute(_numObjects.lineIndices, _geometryBuffers, generalRenderInfo, _simulationFacade);
        }
        currentRenderStep = nextRenderStep;
    } while (currentRenderStep);
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
