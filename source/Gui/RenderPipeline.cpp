#include "RenderPipeline.h"

#include "EngineInterface/GeometryBuffers.h"

#include "RenderStep.h"
#include "RenderStep.h"
#include "Shader.h"

_RenderPipeline::_RenderPipeline(SimulationFacade const& simulationFacade)
    : _simulationFacade(simulationFacade)
    , _geometryBuffers(_GeometryBuffers::create())
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
}

void _RenderPipeline::addStep(RenderStep const& step)
{
    _steps.emplace_back(step);
}

void _RenderPipeline::finalize()
{
    std::vector<RenderStep> stepsWithoutDependencies;
    for (auto const& step : _steps) {
        if (step->getDependentSteps().empty()) {
            stepsWithoutDependencies.emplace_back(step);
        }
    }
    CHECK(!stepsWithoutDependencies.empty());

    auto currentRenderStep = stepsWithoutDependencies.front();
    std::set<RenderStep> finishedSteps;
    do {
        finishedSteps.insert(currentRenderStep);
        auto nextRenderStep = findNextStep(finishedSteps);
        if (!currentRenderStep->getTarget().has_value()) {
            auto finalStep = nextRenderStep == nullptr;
            if (finalStep) {
                currentRenderStep->setTarget(ScreenTarget());
            } else {
                currentRenderStep->setTarget(_TextureTarget::create());
            }
        }
        currentRenderStep = nextRenderStep;
    } while (currentRenderStep);

    std::set<RenderTarget> initialTargets;
    for (auto const& step : stepsWithoutDependencies) {
        initialTargets.insert(step->getTarget().value());
    }

    for (auto const& step : stepsWithoutDependencies) {
        if (initialTargets.contains(step->getTarget().value())) {
            step->setClearBackground(true);
            initialTargets.erase(step->getTarget().value());
        }
    }
}

void _RenderPipeline::resize(IntVector2D const& size)
{
    std::set<TextureTarget> textureTargets;

    for (auto const& step : _steps) {
        if (std::holds_alternative<TextureTarget>(step->getTarget().value())) {
            textureTargets.insert(std::get<TextureTarget>(step->getTarget().value()));
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
    std::vector<RenderStep> stepsWithoutDependencies;
    for (auto const& step : _steps) {
        if (step->getDependentSteps().empty()) {
            stepsWithoutDependencies.emplace_back(step);
        }
    }
    CHECK(!stepsWithoutDependencies.empty());
    auto currentRenderStep = stepsWithoutDependencies.front();

    // Copy vertex buffer from Cuda to OpenGL
    auto numRenderObjects = _simulationFacade->tryCopyBuffersFromCudaToOpenGL(_geometryBuffers);
    if (numRenderObjects.has_value()) {
        _numObjects = *numRenderObjects;
    }

    GeneralRenderInfo generalRenderInfo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &generalRenderInfo.screenFbo);

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
        if (auto triangleRenderStep = std::dynamic_pointer_cast<_TriangleRenderStep>(currentRenderStep)) {
            triangleRenderStep->execute(_numObjects.triangleIndices, _geometryBuffers, generalRenderInfo, _simulationFacade);
        }
        if (auto postProcessingRenderStep = std::dynamic_pointer_cast<_PostProcessingRenderStep>(currentRenderStep)) {
            postProcessingRenderStep->execute(generalRenderInfo, _simulationFacade);
        }
        currentRenderStep = nextRenderStep;
    } while (currentRenderStep);

    glBindFramebuffer(GL_FRAMEBUFFER, generalRenderInfo.screenFbo);
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
