#include "RenderStep.h"

#include "EngineInterface/RenderData.h"
#include "EngineInterface/SimulationFacade.h"

#include "Shader.h"
#include "Viewport.h"

_RenderStep::_RenderStep(Shader const& shader, TargetData const& targetData, std::vector<RenderStep> const& dependentSteps)
    : _dependentSteps(dependentSteps)
    , _shader(shader)
    , _targetData(targetData)
    , _sharedTarget(targetData != nullptr)
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
    if (_sharedTarget) {
        return;
    }
    if (_targetData != nullptr) {
        glDeleteFramebuffers(1, &_targetData->fbo);
        glDeleteTextures(1, &_targetData->outputTexture);
    } else {
        _targetData = std::make_shared<_TargetData>();
    }
    // Init output texture
    glGenTextures(1, &_targetData->outputTexture);
    glBindTexture(GL_TEXTURE_2D, _targetData->outputTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    // Init framebuffer
    glGenFramebuffers(1, &_targetData->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _targetData->fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _targetData->outputTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

std::vector<RenderStep> const& _RenderStep::getDependentSteps() const
{
    return _dependentSteps;
}

Shader _RenderStep::getShader() const
{
    return _shader;
}

TargetData _RenderStep::getTargetData() const
{
    return _targetData;
}

unsigned int _RenderStep::getTexture() const
{
    return _targetData->outputTexture;
}

unsigned int _RenderStep::getFbo() const
{
    return _targetData->fbo;
}

void _RenderStep::setBool(std::string const& name, bool value)
{
    _boolValues.insert_or_assign(name, value);
}

_PointRenderStep::_PointRenderStep(Shader const& shader)
    : _RenderStep(shader, nullptr, {})
{
    auto vao = _shader->getVao();
    auto vbo = _shader->getVbo();

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

PointRenderStep _PointRenderStep::create(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader)
{
    auto shader = std::make_shared<_Shader>(vertexShader, fragmentShader);
    return PointRenderStep(new _PointRenderStep(shader));
}

PointRenderStep _PointRenderStep::createWithSharedVbo(
    std::filesystem::path const& vertexShader,
    std::filesystem::path const& fragmentShader,
    RenderStep const& sharedStep)
{
    auto shader = std::make_shared<_Shader>(vertexShader, fragmentShader, std::filesystem::path(), sharedStep->getShader()->getVbo());
    return PointRenderStep(new _PointRenderStep(shader));
}

PostProcessingRenderStep _PostProcessingRenderStep::create(
    std::filesystem::path const& vertexShader,
    std::filesystem::path const& fragmentShader,
    std::vector<RenderStep> const& dependentSteps)
{
    auto shader = std::make_shared<_Shader>(vertexShader, fragmentShader);
    return PostProcessingRenderStep(new _PostProcessingRenderStep(shader, dependentSteps));
}

void _PointRenderStep::execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade)
{
    auto viewSize = Viewport::get().getViewSize();

    if (std::holds_alternative<TextureTarget>(target)) {
        glBindFramebuffer(GL_FRAMEBUFFER, _targetData->fbo);
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

LineRenderStep _LineRenderStep::createWithSharedVboAndTarget(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, RenderStep const& sharedStep)
{
    auto shader = std::make_shared<_Shader>(vertexShader, fragmentShader, std::filesystem::path(), sharedStep->getShader()->getVbo());
    return LineRenderStep(new _LineRenderStep(shader, sharedStep->getTargetData()));
}

_LineRenderStep::_LineRenderStep(Shader const& shader, TargetData const& targetData)
    : _RenderStep(shader, targetData, {})
{
    auto vao = _shader->getVao();
    auto vbo = _shader->getVbo();
    auto ebo = _shader->getEbo();

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

void _LineRenderStep::execute(RenderTarget const& target, NumRenderObjects const& numObjects, SimulationFacade const& simulationFacade)
{
    //auto viewSize = Viewport::get().getViewSize();

    //// Render to the same framebuffer as the shared step (point renderer)
    //glBindFramebuffer(GL_FRAMEBUFFER, _sharedStep->getFbo());
    //glViewport(0, 0, viewSize.x, viewSize.y);

    //// Enable blending for anti-aliasing
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    //activateShader(simulationFacade);

    //// Draw lines
    //glBindVertexArray(_shader->getVao());
    //glDrawElements(GL_LINES, toInt(numObjects.lineIndices), GL_UNSIGNED_INT, 0);

    //// Disable blending
    //glDisable(GL_BLEND);
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
        glBindFramebuffer(GL_FRAMEBUFFER, _targetData->fbo);
    } else {
        auto fbo = std::get<ScreenTarget>(target).fbo;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    }
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

_PostProcessingRenderStep::_PostProcessingRenderStep(Shader const& shader, std::vector<RenderStep> const& dependentSteps)
    : _RenderStep(shader, nullptr, dependentSteps)
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
