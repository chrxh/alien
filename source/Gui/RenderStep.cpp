#include "RenderStep.h"

#include "EngineInterface/SimulationFacade.h"
#include "EngineInterface/RenderData.h"

#include "Shader.h"
#include "Viewport.h"

_RenderStep::_RenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, std::vector<RenderStep> const& dependentSteps)
    : _dependentSteps(dependentSteps)
{
    _shader = std::make_shared<_Shader>(vertexShader, fragmentShader);
}

_PointRenderStep::_PointRenderStep(
    std::filesystem::path const& vertexShader,
    std::filesystem::path const& fragmentShader,
    std::vector<RenderStep> const& dependentSteps,
    SimulationFacade const& simulationFacade)
    : _RenderStep(vertexShader, fragmentShader, dependentSteps)
    , _simulationFacade(simulationFacade)
{
    auto vao = _shader->getVao();
    auto vbo = _shader->getVbo();

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 1000000 * sizeof(VertexData), nullptr, GL_DYNAMIC_DRAW);

    // Setup vertex attributes for RenderingObjectData
    // Position (2 floats)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)0);
    glEnableVertexAttribArray(0);

    // Color (3 floats)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void _PointRenderStep::resize()
{
}

void _PointRenderStep::execute(NumRenderObjects const& numObjects)
{
    auto worldSize = _simulationFacade->getWorldSize();
    auto worldRect = Viewport::get().getVisibleWorldRect();
    auto viewSize = Viewport::get().getViewSize();
    auto zoomFactor = Viewport::get().getZoomFactor();

    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
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

    // Use background object shader
    _shader->use();
    _shader->setFloat("zoom", zoomFactor);
    _shader->setFloat("radius", std::max(6.0f, zoomFactor));  // std::max to avoid moire patterns at low zoom factors
    _shader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
    _shader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
    _shader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));

    // Draw points
    glBindVertexArray(_shader->getVao());
    glDrawArrays(GL_POINTS, 0, toInt(numObjects.vertices));
}

_PostProcessingRenderStep::_PostProcessingRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader)
    : _RenderStep(vertexShader, fragmentShader, {})
{
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

    auto vao = _shader->getVao();
    auto vbo = _shader->getVbo();
    auto ebo = _shader->getEbo();
    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void _PostProcessingRenderStep::resize() {}

void _PostProcessingRenderStep::execute(NumRenderObjects const& numObjects) {}
