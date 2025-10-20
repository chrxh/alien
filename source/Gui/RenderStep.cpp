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

_RenderStep::_RenderStep(StepParameters const& parameters)
    : _previousTargetSelection(parameters._previousTargetSelection)
    , _textureScale(parameters._textureScale)
    , _uniforms(parameters._uniforms)
    , _uniformFunc(parameters._uniformFunc)
    , _preventMoirePatterns(parameters._preventMoirePatterns)
{
    if (!parameters._shader.empty()) {
        auto vertexShaderPath = parameters._shader;
        vertexShaderPath.replace_extension(".vs");
        auto fragmentShaderPath = parameters._shader;
        fragmentShaderPath.replace_extension(".fs");
        auto geometryShaderPath = parameters._shader;
        geometryShaderPath.replace_extension(".gs");

        if (!std::filesystem::exists(geometryShaderPath)) {
            geometryShaderPath = std::filesystem::path();  // empty path disables geometry shader
        }

        _shader = _Shader::create(vertexShaderPath, fragmentShaderPath, geometryShaderPath);
    }
}

std::optional<int> const& _RenderStep::getPreviousTargetSelection() const
{
    return _previousTargetSelection;
}

TextureTarget const& _RenderStep::getTextureTarget() const
{
    return _target;
}

void _RenderStep::setTextureTarget(TextureTarget const& target)
{
    _target = target;
}

void _RenderStep::resize(IntVector2D const& size)
{
    if (_target->initialized) {
        glDeleteFramebuffers(1, &_target->fbo);
        glDeleteTextures(1, &_target->texture);
    }
    // Init output texture
    glGenTextures(1, &_target->texture);
    glBindTexture(GL_TEXTURE_2D, _target->texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, size.x, size.y, 0, GL_RGBA, GL_FLOAT, NULL);

    // Init framebuffer
    glGenFramebuffers(1, &_target->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _target->fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _target->texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

float _RenderStep::getTextureScale() const
{
    return _textureScale;
}

void _RenderStep::setTextureScale(float scale)
{
    _textureScale = scale;
}

void _RenderStep::prepareExecution(ExecutionParameters const& parameters)
{
    auto worldSize = parameters._simulationFacade->getWorldSize();
    auto worldRect = Viewport::get().getVisibleWorldRect();
    auto viewSize = Viewport::get().getViewSize();
    auto zoom = Viewport::get().getZoomFactor();
    //auto timestep = simulationFacade->getCurrentTimestep();

    _shader->use();
    _shader->setFloat("zoom", zoom);
    _shader->setFloat("radius", _preventMoirePatterns ? std::max(6.0f, zoom) : zoom);
    _shader->setVec2("worldSize", toRealVector2D(worldSize));
    _shader->setVec2("rectUpperLeft", worldRect.topLeft);
    _shader->setVec2("viewportSize", toRealVector2D(viewSize));
    //_shader->setFloat("lightAngle", toFloat(timestep % 10000) / 10000.0f * 360.0f);

    auto uniforms = _uniforms;
    if (_uniformFunc) {
        auto uniformFunc = _uniformFunc();
        uniforms.insert(uniformFunc.begin(), uniformFunc.end());
    }
    for (auto const& [key, value] : uniforms) {
        if (std::holds_alternative<int>(value)) {
            _shader->setInt(key, std::get<int>(value));
        }
        if (std::holds_alternative<float>(value)) {
            _shader->setFloat(key, std::get<float>(value));
        }
        if (std::holds_alternative<FloatColorRGB>(value)) {
            _shader->setVec3(key, std::get<FloatColorRGB>(value));
        }
    }

    if (std::holds_alternative<ScreenTarget>(parameters._target)) {
        glBindFramebuffer(GL_FRAMEBUFFER, parameters._renderInfo.screenFbo);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, std::get<TextureTarget>(parameters._target)->fbo);
    }
    glViewport(0, 0, toInt(toFloat(viewSize.x) * parameters._scale), toInt(toFloat(viewSize.y) * parameters._scale));

    if (parameters._clearBackground) {
        // Clear with black background
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

CellRenderStep _CellRenderStep::create(StepParameters const& parameters)
{
    return CellRenderStep(new _CellRenderStep(parameters));
}

void _CellRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Draw points
    glBindVertexArray(parameters._geometryBuffers->getVaoForPointsAndLines());
    glDrawArrays(GL_POINTS, 0, toInt(parameters._geometryBuffers->getNumObjects().vertices));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_CellRenderStep::_CellRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

LineRenderStep _LineRenderStep::create(StepParameters const& parameters)
{
    return LineRenderStep(new _LineRenderStep(parameters));
}

void _LineRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);
    
    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Draw lines (geometry shader will convert to quads with proper width)
    glBindVertexArray(parameters._geometryBuffers->getVaoForPointsAndLines());
    glDrawElements(GL_LINES, toInt(parameters._geometryBuffers->getNumObjects().lineIndices), GL_UNSIGNED_INT, 0);
    
    // Disable blending
    glDisable(GL_BLEND);
}

_LineRenderStep::_LineRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{
}

TriangleRenderStep _TriangleRenderStep::create(StepParameters const& parameters)
{
    return TriangleRenderStep(new _TriangleRenderStep(parameters));
}

void _TriangleRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(/*GL_SRC_ALPHA*/ GL_ONE, /*GL_ONE*/ GL_ZERO);

    // Draw triangles
    glBindVertexArray(parameters._geometryBuffers->getVaoForTriangles());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, parameters._geometryBuffers->getEboForTriangles());
    glDrawElements(GL_TRIANGLES, toInt(parameters._geometryBuffers->getNumObjects().triangleIndices), GL_UNSIGNED_INT, 0);
    
    // Disable blending
    glDisable(GL_BLEND);
}

_TriangleRenderStep::_TriangleRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{
}

PostProcessingRenderStep _PostProcessingRenderStep::create(StepParameters const& parameters)
{
    return PostProcessingRenderStep(new _PostProcessingRenderStep(parameters));
}

void _PostProcessingRenderStep::execute(ExecutionParameters parameters)
{
    parameters._clearBackground = false;
    prepareExecution(parameters);

    glBindVertexArray(_vao);
    
    auto numTextures = parameters._textures.size();
    CHECK(numTextures <= 3);
    _shader->setInt("numTextures", toInt(numTextures));
    if (numTextures >= 1) {
        _shader->setInt("inputTexture1", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, parameters._textures.at(0));
    }
    if (numTextures >= 2) {
        _shader->setInt("inputTexture2", 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, parameters._textures.at(1));
    }
    if (numTextures >= 3) {
        _shader->setInt("inputTexture3", 2);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, parameters._textures.at(2));
    }
    
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

_PostProcessingRenderStep::_PostProcessingRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
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

ForwardRenderStep _ForwardRenderStep::create(StepParameters const& parameters)
{
    return ForwardRenderStep(new _ForwardRenderStep(parameters));
}

void _ForwardRenderStep::execute(ExecutionParameters parameters)
{
    // Do nothing
}

_ForwardRenderStep::_ForwardRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{
}

EnergyParticleRenderStep _EnergyParticleRenderStep::create(StepParameters const& parameters)
{
    return EnergyParticleRenderStep(new _EnergyParticleRenderStep(parameters));
}

void _EnergyParticleRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    // Enable additive blending for energy particles
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Draw energy particles
    glBindVertexArray(parameters._geometryBuffers->getVaoForEnergyParticles());
    glDrawArrays(GL_POINTS, 0, toInt(parameters._geometryBuffers->getNumObjects().energyParticles));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_EnergyParticleRenderStep::_EnergyParticleRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

LocationRenderStep _LocationRenderStep::create(StepParameters const& parameters)
{
    return LocationRenderStep(new _LocationRenderStep(parameters));
}

void _LocationRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    // Enable blending for semi-transparent locations
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw location points
    glBindVertexArray(parameters._geometryBuffers->getVaoForLocations());
    glDrawArrays(GL_POINTS, 0, toInt(parameters._geometryBuffers->getNumObjects().locations));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_LocationRenderStep::_LocationRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}
