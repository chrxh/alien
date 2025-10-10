#include "RenderStep.h"

#include "EngineInterface/RenderData.h"
#include "EngineInterface/SimulationFacade.h"

#include "Shader.h"
#include "Viewport.h"

_RenderStep::_RenderStep(Shader const& shader, std::vector<RenderStep> const& dependentSteps)
    : _dependentSteps(dependentSteps)
    , _shader(shader)
{
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

    for (auto const& [key, value] : _boolValues) {
        _shader->setBool(key, value);
    }
}

void _RenderStep::resize(IntVector2D const& size)
{
    if (!_outputTextureInitialized) {
        glDeleteFramebuffers(1, &_fbo);
        glDeleteTextures(1, &_outputTexture);
    }
    // Init output texture
    glGenTextures(1, &_outputTexture);
    glBindTexture(GL_TEXTURE_2D, _outputTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    // Init framebuffer
    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _outputTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    _outputTextureInitialized = true;
}

std::vector<RenderStep> const& _RenderStep::getDependentSteps() const
{
    return _dependentSteps;
}

Shader _RenderStep::getShader() const
{
    return _shader;
}

unsigned int _RenderStep::getTexture() const
{
    return _outputTexture;
}

void _RenderStep::setBool(std::string const& name, bool value)
{
    _boolValues.insert_or_assign(name, value);
}

_AbstractPointRenderStep::_AbstractPointRenderStep(Shader const& shader)
    : _RenderStep(shader, {})
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

_PointRenderStep::_PointRenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader)
    : _AbstractPointRenderStep(std::make_shared<_Shader>(vertexShader, fragmentShader))
{}

_SharedVboPointRenderStep::_SharedVboPointRenderStep(
    std::filesystem::path const& vertexShader,
    std::filesystem::path const& fragmentShader,
    RenderStep const& sharedStep)
    : _AbstractPointRenderStep(std::make_shared<_Shader>(vertexShader, fragmentShader, std::filesystem::path(), sharedStep->getShader()->getVbo()))
{}

void _AbstractPointRenderStep::execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade)
{
    auto viewSize = Viewport::get().getViewSize();

    if (std::holds_alternative<TextureTarget>(target)) {
        glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    } else {
        auto fbo = std::get<ScreenTarget>(target).fbo;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
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
    glBindVertexArray(_shader->getVao());
    glDrawArrays(GL_POINTS, 0, toInt(numObjects.vertices));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_PostProcessingRenderStep::_PostProcessingRenderStep(
    std::filesystem::path const& vertexShader,
    std::filesystem::path const& fragmentShader,
    std::vector<RenderStep> const& dependentSteps)
    : _RenderStep(std::make_shared<_Shader>(vertexShader, fragmentShader), dependentSteps)
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

void _PostProcessingRenderStep::execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade)
{
    activateShader(simulationFacade);

    glBindVertexArray(_shader->getVao());

    // Input
    auto numDependentSteps = _dependentSteps.size();
    CHECK(numDependentSteps <= 2);
    if (numDependentSteps >= 1) {
        _shader->setInt("inputTexture1", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _dependentSteps.at(0)->getTexture());
    }
    if (numDependentSteps >= 2) {
        _shader->setInt("inputTexture2", 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, _dependentSteps.at(1)->getTexture());
    }

    // Output
    if (std::holds_alternative<TextureTarget>(target)) {
        glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    } else {
        auto fbo = std::get<ScreenTarget>(target).fbo;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    }
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

