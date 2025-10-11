#include "RenderStep2.h"

#include "EngineInterface/SimulationFacade.h"

#include "Shader.h"
#include "Viewport.h"
#include <EngineInterface/RenderData.h>

GeometrySource GeometrySource::create()
{
    GeometrySource result;
    glGenVertexArrays(1, &result.vao);
    glGenBuffers(1, &result.vbo);
    glGenBuffers(1, &result.ebo);
    return result;
}

std::vector<RenderStep2> const& _RenderStep2::getDependentSteps() const
{
    return _dependentSteps;
}

RenderSource const& _RenderStep2::getSource() const
{
    return _source;
}

RenderTarget const& _RenderStep2::getTarget() const
{
    return _target;
}

_RenderStep2::_RenderStep2(Shader const& shader, RenderSource const& source, RenderTarget const& target, std::vector<RenderStep2> const& dependentSteps)
    : _shader(shader)
    , _source(source)
    , _target(target)
    , _dependentSteps(dependentSteps)
{}

void _RenderStep2::activateShader(SimulationFacade const& simulationFacade)
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

PointRenderStep2 _PointRenderStep2::create(Shader const& shader, GeometrySource const& source, RenderTarget const& target)
{
    return PointRenderStep2(new _PointRenderStep2(shader, source, target));
}

void _PointRenderStep2::execute(uint64_t const& numVertices, GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade)
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
    auto geometrySource = std::get<GeometrySource>(_source);
    glBindVertexArray(geometrySource.vao);
    glDrawArrays(GL_POINTS, 0, toInt(numVertices));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_PointRenderStep2::_PointRenderStep2(Shader const& shader, GeometrySource const& source, RenderTarget const& target)
    : _RenderStep2(shader, source, target, {})
{
        auto vao = source.vao;
        auto vbo = source.vbo;
    
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
        // Setup vertex attributes for RenderingObjectData
        // Position (2 floats)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);
        glEnableVertexAttribArray(0);
    
        // Color (3 floats)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
}
