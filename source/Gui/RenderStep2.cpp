#include "RenderStep2.h"

#include "EngineInterface/SimulationFacade.h"

#include "RenderPipeline.h"
#include <EngineInterface/RenderData.h>

#include "Shader.h"
#include "Viewport.h"

std::vector<RenderStep> const& _RenderStep::getDependentSteps() const
{
    return _dependentSteps;
}

RenderTarget const& _RenderStep::getTarget() const
{
    return _target;
}

_RenderStep::_RenderStep(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps)
    : _shader(shader)
    , _target(target)
    , _dependentSteps(dependentSteps)
{}

void _RenderStep::activateShader(SimulationFacade const& simulationFacade)
{
    auto worldSize = simulationFacade->getWorldSize();
    auto worldRect = Viewport::get().getVisibleWorldRect();
    auto viewSize = Viewport::get().getViewSize();
    auto zoom = Viewport::get().getZoomFactor();

    _shader->use();
    _shader->setFloat("zoom", zoom);
    _shader->setFloat("radius", std::max(6.0f, zoom));  // std::max to avoid moire patterns at low zoom factors
    _shader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
    _shader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
    _shader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
}

PointRenderStep _PointRenderStep::create(Shader const& shader,RenderTarget const& target)
{
    return PointRenderStep(new _PointRenderStep(shader, target));
}

void _PointRenderStep::execute(
    uint64_t const& numVertices,
    GeometrySource const& geometrySource,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    auto viewSize = Viewport::get().getViewSize();

    if (std::holds_alternative<TextureTarget>(_target)) {
        auto textureTarget = std::get<TextureTarget>(_target);
        glBindFramebuffer(GL_FRAMEBUFFER, textureTarget->fbo);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, renderInfo.screenFbo);
    }
    glViewport(0, 0, viewSize.x, viewSize.y);

    // Clear with black background
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    activateShader(simulationFacade);

    // Draw points
    glBindVertexArray(geometrySource->getVao());
    glDrawArrays(GL_POINTS, 0, toInt(numVertices));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_PointRenderStep::_PointRenderStep(Shader const& shader, RenderTarget const& target)
    : _RenderStep(shader, target, {})
{}

LineRenderStep _LineRenderStep::create(Shader const& shader, RenderTarget const& target, RenderStep const& dependentStep)
{
    return LineRenderStep(new _LineRenderStep(shader, target, dependentStep));
}

void _LineRenderStep::execute(
    uint64_t const& numLines,
    GeometrySource const& geometrySource,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    auto viewSize = Viewport::get().getViewSize();
    auto zoom = Viewport::get().getZoomFactor();

    if (std::holds_alternative<TextureTarget>(_target)) {
        auto textureTarget = std::get<TextureTarget>(_target);
        glBindFramebuffer(GL_FRAMEBUFFER, textureTarget->fbo);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, renderInfo.screenFbo);
    }
    glViewport(0, 0, viewSize.x, viewSize.y);
    
    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    
    activateShader(simulationFacade);
    
    // Draw lines
    glBindVertexArray(geometrySource->getVao());
    glLineWidth(zoom * 0.1f);
    glDrawElements(GL_LINES, toInt(numLines), GL_UNSIGNED_INT, 0);
    
    // Disable blending
    glDisable(GL_BLEND);
}

_LineRenderStep::_LineRenderStep(Shader const& shader, RenderTarget const& target, RenderStep const& dependentStep)
    : _RenderStep(shader, target, {dependentStep})
{}
