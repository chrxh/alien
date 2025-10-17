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

_RenderStep::_RenderStep(
    std::filesystem::path const& shaderFilename,
    std::optional<int> const& previousTargetSelection,
    std::map<std::string, UniformValueType> const& uniformValues)
    : _previousTargetSelection(previousTargetSelection)
    , _uniformValues(uniformValues)
{
    auto vertexShaderPath = shaderFilename;
    vertexShaderPath.replace_extension(".vs");
    auto fragmentShaderPath = shaderFilename;
    fragmentShaderPath.replace_extension(".fs");
    auto geometryShaderPath = shaderFilename;
    geometryShaderPath.replace_extension(".gs");

    if (!std::filesystem::exists(geometryShaderPath)) {
        geometryShaderPath = std::filesystem::path();  // empty path disables geometry shader
    }

    _shader = _Shader::create(vertexShaderPath, fragmentShaderPath, geometryShaderPath);
}

_RenderStep::_RenderStep(bool previousTargetSelection)
    : _previousTargetSelection(previousTargetSelection)
{}

std::optional<int> const& _RenderStep::getPreviousTargetSelection() const
{
    return _previousTargetSelection;
}

std::optional<TextureTarget> const& _RenderStep::getTextureTarget() const
{
    return _target;
}

void _RenderStep::setTextureTarget(TextureTarget const& target)
{
    _target = target;
}

void _RenderStep::prepareExecution(
    bool clearBackground,
    RenderTarget const& target,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    auto worldSize = simulationFacade->getWorldSize();
    auto worldRect = Viewport::get().getVisibleWorldRect();
    auto viewSize = Viewport::get().getViewSize();
    auto zoom = Viewport::get().getZoomFactor();
    //auto timestep = simulationFacade->getCurrentTimestep();

    _shader->use();
    _shader->setFloat("zoom", zoom);
    _shader->setFloat("radius", std::max(6.0f, zoom));  // std::max to avoid moire patterns at low zoom factors
    _shader->setVec2("worldSize", toFloat(worldSize.x), toFloat(worldSize.y));
    _shader->setVec2("rectUpperLeft", worldRect.topLeft.x, worldRect.topLeft.y);
    _shader->setVec2("viewportSize", toFloat(viewSize.x), toFloat(viewSize.y));
    //_shader->setFloat("lightAngle", toFloat(timestep % 10000) / 10000.0f * 360.0f);

    for (auto const& [key, value] : _uniformValues) {
        if (std::holds_alternative<int>(value)) {
            _shader->setInt(key, std::get<int>(value));
        }
        if (std::holds_alternative<float>(value)) {
            _shader->setFloat(key, std::get<float>(value));
        }
    }

    if (std::holds_alternative<ScreenTarget>(target)) {
        glBindFramebuffer(GL_FRAMEBUFFER, renderInfo.screenFbo);
    } else {
        glBindFramebuffer(GL_FRAMEBUFFER, std::get<TextureTarget>(target)->fbo);
    }
    glViewport(0, 0, viewSize.x, viewSize.y);

    if (clearBackground) {
        // Clear with black background
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
    }
}

PointRenderStep _PointRenderStep::create(std::filesystem::path const& shaderFilename, std::optional<int> const& previousTargetSelection)
{
    return PointRenderStep(new _PointRenderStep(shaderFilename, previousTargetSelection));
}

void _PointRenderStep::execute(
    GeometryBuffers const& geometryBuffers,
    std::vector<unsigned> const& textures,
    bool clearBackground,
    RenderTarget const& target,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    prepareExecution(clearBackground, target, renderInfo, simulationFacade);

    // Enable point sprites
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Draw points
    glBindVertexArray(geometryBuffers->getVaoForPointsAndLines());
    glDrawArrays(GL_POINTS, 0, toInt(geometryBuffers->getNumObjects().vertices));

    // Disable blending and point sprites
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}

_PointRenderStep::_PointRenderStep(std::filesystem::path const& shaderFilename, std::optional<int> const& previousTargetSelection)
    : _RenderStep(shaderFilename, previousTargetSelection)
{}

LineRenderStep _LineRenderStep::create(std::filesystem::path const& shaderFilename, std::optional<int> const& previousTargetSelection)
{
    return LineRenderStep(new _LineRenderStep(shaderFilename, previousTargetSelection));
}

void _LineRenderStep::execute(
    GeometryBuffers const& geometryBuffers,
    std::vector<unsigned int> const& textures,
    bool clearBackground,
    RenderTarget const& target,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    prepareExecution(clearBackground, target, renderInfo, simulationFacade);
    
    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Draw lines (geometry shader will convert to quads with proper width)
    glBindVertexArray(geometryBuffers->getVaoForPointsAndLines());
    glDrawElements(GL_LINES, toInt(geometryBuffers->getNumObjects().lineIndices), GL_UNSIGNED_INT, 0);
    
    // Disable blending
    glDisable(GL_BLEND);
}

_LineRenderStep::_LineRenderStep(std::filesystem::path const& shaderFilename, std::optional<int> const& previousTargetSelection)
    : _RenderStep(shaderFilename, previousTargetSelection)
{
}

TriangleRenderStep _TriangleRenderStep::create(std::filesystem::path const& shaderFilename, std::optional<int> const& previousTargetSelection)
{
    return TriangleRenderStep(new _TriangleRenderStep(shaderFilename, previousTargetSelection));
}

void _TriangleRenderStep::execute(
    GeometryBuffers const& geometryBuffers,
    std::vector<unsigned int> const& textures,
    bool clearBackground,
    RenderTarget const& target,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    prepareExecution(clearBackground, target, renderInfo, simulationFacade);

    // Enable blending for anti-aliasing
    glEnable(GL_BLEND);
    glBlendFunc(/*GL_SRC_ALPHA*/ GL_ONE, /*GL_ONE*/ GL_ZERO);

    // Draw triangles
    glBindVertexArray(geometryBuffers->getVaoForTriangles());
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, geometryBuffers->getEboForTriangles());
    glDrawElements(GL_TRIANGLES, toInt(geometryBuffers->getNumObjects().triangleIndices), GL_UNSIGNED_INT, 0);
    
    // Disable blending
    glDisable(GL_BLEND);
}

_TriangleRenderStep::_TriangleRenderStep(std::filesystem::path const& shaderFilename, std::optional<int> const& previousTargetSelection)
    : _RenderStep(shaderFilename, previousTargetSelection)
{
}

PostProcessingRenderStep _PostProcessingRenderStep::create(
    std::filesystem::path const& shaderFilename,
    std::optional<int> const& previousTargetSelection,
    std::map<std::string, UniformValueType> const& uniformValues)
{
    return PostProcessingRenderStep(new _PostProcessingRenderStep(shaderFilename, previousTargetSelection, uniformValues));
}

void _PostProcessingRenderStep::execute(
    GeometryBuffers const& geometryBuffers,
    std::vector<unsigned int> const& textures,
    bool clearBackground,
    RenderTarget const& target,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    prepareExecution(clearBackground, target, renderInfo, simulationFacade);

    glBindVertexArray(_vao);
    
    auto numTextures = textures.size();
    CHECK(numTextures <= 2);
    if (numTextures >= 1) {
        _shader->setInt("inputTexture1", 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textures.at(0));
    }
    if (numTextures >= 2) {
        _shader->setInt("inputTexture2", 1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, textures.at(1));
    }
    
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

_PostProcessingRenderStep::_PostProcessingRenderStep(
    std::filesystem::path const& shaderFilename,
    std::optional<int> const& previousTargetSelection,
    std::map<std::string, UniformValueType> const& uniformValues)
    : _RenderStep(shaderFilename, previousTargetSelection, uniformValues)
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

ForwardRenderStep _ForwardRenderStep::create(int previousTargetSelection)
{
    return ForwardRenderStep(new _ForwardRenderStep(previousTargetSelection));
}

void _ForwardRenderStep::execute(
    GeometryBuffers const& geometryBuffers,
    std::vector<unsigned int> const& textures,
    bool clearBackground,
    RenderTarget const& target,
    GeneralRenderInfo const& renderInfo,
    SimulationFacade const& simulationFacade)
{
    // Do nothing
}

_ForwardRenderStep::_ForwardRenderStep(int previousTargetSelection)
    : _RenderStep(previousTargetSelection)
{
}
