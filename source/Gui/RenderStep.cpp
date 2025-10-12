#include "RenderStep.h"

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/GeometryBuffers.h"

#include "RenderPipeline.h"

#include "Shader.h"
#include "Viewport.h"

TextureTarget _TextureTarget::create()
{
    return TextureTarget(new _TextureTarget());
}

std::vector<RenderStep> const& _RenderStep::getDependentSteps() const
{
    return _dependentSteps;
}


_RenderStep::_RenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps)
    : _shader(shader)
    , _target(target)
    , _dependentSteps(dependentSteps)
{}

std::optional<RenderTarget> const& _RenderStep::getTarget() const
{
    return _target;
}

void _RenderStep::setTarget(RenderTarget const& target)
{
    _target = target;
}

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

    for (auto const& [key, value] : _uniformValues) {
        if (std::holds_alternative<int>(value)) {
            _shader->setInt(key, std::get<int>(value));
        }
    }
}

void _RenderStep::setFramebuffer(GeneralRenderInfo const& renderInfo)
{
    if (std::holds_alternative<TextureTarget>(_target.value())) {
        auto textureTarget = std::get<TextureTarget>(_target.value());
        glBindFramebuffer(GL_FRAMEBUFFER, textureTarget->fbo);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, renderInfo.screenFbo);
    }
}

PointRenderStep _PointRenderStep::create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps)
{
    return PointRenderStep(new _PointRenderStep(shader, target, dependentSteps));
}

PointRenderStep _PointRenderStep::create(Shader const& shader, std::vector<RenderStep> const& dependentSteps)
{
    return PointRenderStep(new _PointRenderStep(shader, std::nullopt, dependentSteps));
}

void _PointRenderStep::execute(
    uint64_t const& numVertices,
    GeometryBuffers const& geometryBuffers,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    auto viewSize = Viewport::get().getViewSize();

    setFramebuffer(renderInfo);
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
    glBindVertexArray(geometryBuffers->getVao());
    glDrawArrays(GL_POINTS, 0, toInt(numVertices));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_PointRenderStep::_PointRenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps)
    : _RenderStep(shader, target, dependentSteps)
{}

LineRenderStep _LineRenderStep::create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps)
{
    return LineRenderStep(new _LineRenderStep(shader, target, dependentSteps));
}

LineRenderStep _LineRenderStep::create(Shader const& shader, std::vector<RenderStep> const& dependentSteps)
{
    return LineRenderStep(new _LineRenderStep(shader, std::nullopt, dependentSteps));
}

void _LineRenderStep::execute(
    uint64_t const& numLines,
    GeometryBuffers const& geometryBuffers,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    auto viewSize = Viewport::get().getViewSize();
    auto zoom = Viewport::get().getZoomFactor();

    setFramebuffer(renderInfo);
    glViewport(0, 0, viewSize.x, viewSize.y);
    
    // Clear with black background
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    
    activateShader(simulationFacade);
    
    // Draw lines
    glBindVertexArray(geometryBuffers->getVao());
    glLineWidth(zoom * 0.1f);
    glDrawElements(GL_LINES, toInt(numLines), GL_UNSIGNED_INT, 0);
    
    // Disable blending
    glDisable(GL_BLEND);
}

_LineRenderStep::_LineRenderStep(Shader const& shader, std::optional<RenderTarget> const& target, std::vector<RenderStep> const& dependentSteps)
    : _RenderStep(shader, target, dependentSteps)
{
}

PostProcessingRenderStep _PostProcessingRenderStep::create(Shader const& shader, RenderTarget const& target, std::vector<RenderStep> const& dependentSteps)
{
    return PostProcessingRenderStep(new _PostProcessingRenderStep(shader, target, dependentSteps));
}

PostProcessingRenderStep _PostProcessingRenderStep::create(Shader const& shader, std::vector<RenderStep> const& dependentSteps)
{
    return PostProcessingRenderStep(new _PostProcessingRenderStep(shader, std::nullopt, dependentSteps));
}

void _PostProcessingRenderStep::execute(GeneralRenderInfo const& renderInfo, SimulationFacade const& simulationFacade)
{
        activateShader(simulationFacade);
    
        glBindVertexArray(_vao);
    
        // Input
        auto numDependentSteps = _dependentSteps.size();
        CHECK(numDependentSteps <= 2);
        if (numDependentSteps >= 1) {
            auto textureTarget = std::get<TextureTarget>(_dependentSteps.at(0)->getTarget().value());

            _shader->setInt("inputTexture1", 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureTarget->texture);
        }
        if (numDependentSteps >= 2) {
            auto textureTarget = std::get<TextureTarget>(_dependentSteps.at(1)->getTarget().value());

            _shader->setInt("inputTexture2", 1);
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, textureTarget->texture);
        }
    
        // Output
        if (std::holds_alternative<TextureTarget>(_target.value())) {
            auto textureTarget = std::get<TextureTarget>(_target.value());
            glBindFramebuffer(GL_FRAMEBUFFER, textureTarget->fbo);
        } else {
            glBindFramebuffer(GL_FRAMEBUFFER, renderInfo.screenFbo);
        }
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

_PostProcessingRenderStep::_PostProcessingRenderStep(
    Shader const& shader,
    std::optional<RenderTarget> const& target,
    std::vector<RenderStep> const& dependentSteps)
    : _RenderStep(shader, target, dependentSteps)
{
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_ebo);

    // Setup full-screen quad
    float vertices[] = {
        1.0f,  1.0f,  0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 0.0f, 1.0f   // top left
    };
    unsigned int indices[] = {
        0,
        1,
        3,  // first triangle
        1,
        2,
        3  // second triangle
    };

    glBindVertexArray(_vao);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}
