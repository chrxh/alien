#include "RenderStep.h"

#include "Base/Math.h"
#include "EngineInterface/GeometryBuffers.h"
#include "EngineInterface/SimulationFacade.h"

#include "RenderPipeline.h"
#include "Shader.h"
#include "SimulationView.h"
#include "StyleRepository.h"
#include "Viewport.h"

namespace
{
    auto constexpr ZoomFactorForCellDetails = 25.0f;    // Cell type strings and arrows
}

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
        CHECK(std::filesystem::exists(vertexShaderPath));
        CHECK(std::filesystem::exists(fragmentShaderPath));
        _shader = _Shader::create(vertexShaderPath, fragmentShaderPath, geometryShaderPath);
    }
}

std::optional<int> const& _RenderStep::getPreviousTargetSelection() const
{
    return _previousTargetSelection;
}

float _RenderStep::getTextureScaling() const
{
    return _textureScale;
}

void _RenderStep::setTextureScaling(float scale)
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
    _shader->setVec2("rectLowerRight", worldRect.bottomRight);
    _shader->setVec2("viewportSize", toRealVector2D(viewSize));
    //_shader->setFloat("lightAngle", toFloat(timestep % 10000) / 10000.0f * 360.0f);

    auto uniforms = _uniforms;
    if (_uniformFunc) {
        auto uniformFunc = _uniformFunc(*parameters._simulationParameters);
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
    glViewport(0, 0, toInt(toFloat(viewSize.x) * _textureScale), toInt(toFloat(viewSize.y) * _textureScale));

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
{}

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
{}

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
{}

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

    // Enable blending for semi-transparent locations
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw location points (geometry shader will convert to quads)
    glBindVertexArray(parameters._geometryBuffers->getVaoForLocations());
    glDrawArrays(GL_POINTS, 0, toInt(parameters._geometryBuffers->getNumObjects().locations));

    // Disable blending
    glDisable(GL_BLEND);
}

_LocationRenderStep::_LocationRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

SelectedObjectRenderStep _SelectedObjectRenderStep::create(StepParameters const& parameters)
{
    return SelectedObjectRenderStep(new _SelectedObjectRenderStep(parameters));
}

void _SelectedObjectRenderStep::execute(ExecutionParameters parameters)
{
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // Enable blending for semi-transparent circles
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Draw selected objects (cells and energy particles) as points (geometry shader will convert to quads)
    glBindVertexArray(parameters._geometryBuffers->getVaoForSelectedObjects());
    glDrawArrays(GL_POINTS, 0, toInt(parameters._geometryBuffers->getNumObjects().selectedObjects));

    // Disable blending
    glDisable(GL_BLEND);
}

_SelectedObjectRenderStep::_SelectedObjectRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

CellTypeOverlayRenderStep _CellTypeOverlayRenderStep::create(StepParameters const& parameters)
{
    return CellTypeOverlayRenderStep(new _CellTypeOverlayRenderStep(parameters));
}

_CellTypeOverlayRenderStep::~_CellTypeOverlayRenderStep()
{
    if (_cellTypeTextureAtlas != 0) {
        glDeleteTextures(1, &_cellTypeTextureAtlas);
        _cellTypeTextureAtlas = 0;
    }
}

void _CellTypeOverlayRenderStep::execute(ExecutionParameters parameters)
{
    // Only render if zoom exceeds threshold and overlay is active
    auto zoom = Viewport::get().getZoomFactor();
    auto overlayActive = SimulationView::get().isOverlayActive();

    if (zoom <= ZoomFactorForCellDetails || !overlayActive) {
        return;
    }

    // Don't clear background - we want to composite on top of existing rendering
    parameters._clearBackground = false;
    prepareExecution(parameters);

    // Enable blending for semi-transparent overlay
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Bind overlay texture
    _shader->setInt("overlayTexture", 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _cellTypeTextureAtlas);

    // Draw overlay points (geometry shader will convert to textured quads)
    glBindVertexArray(parameters._geometryBuffers->getVaoForPointsAndLines());
    glDrawArrays(GL_POINTS, 0, toInt(parameters._geometryBuffers->getNumObjects().vertices));

    // Disable blending
    glDisable(GL_BLEND);
}

_CellTypeOverlayRenderStep::_CellTypeOverlayRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{
    createCellTypeTextureAtlas();
}

void _CellTypeOverlayRenderStep::createCellTypeTextureAtlas()
{
    // Create a texture atlas containing all cell type strings
    // We'll arrange them in a vertical strip, one per row
    auto font = StyleRepository::get().getDefaultFont();
    float fontSize = 16.0f;  // Base font size for rendering

    // Calculate dimensions for each cell type label
    std::vector<ImVec2> textSizes;

    for (auto const& cellTypeStr : Const::CellTypeStrings) {
        auto textSize = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, cellTypeStr.c_str());
        textSizes.push_back(textSize);
    }

    int textureWidth = 512;   // Fixed width
    int textureHeight = 512;  // Fixed height, should be enough for all labels

    // Create pixel buffer (clear to transparent)
    std::vector<uint8_t> pixels(textureWidth * textureHeight * 4, 0);  // RGBA

    // Get font atlas data
    int atlasWidth, atlasHeight;
    unsigned char* atlasData;
    font->ContainerAtlas->GetTexDataAsAlpha8(&atlasData, &atlasWidth, &atlasHeight);

    // Render each cell type string to the buffer using ImGui font
    int rowHeight = 20;
    float scale = fontSize / font->FontSize;

    for (size_t i = 0; i < Const::CellTypeStrings.size(); ++i) {
        auto const& cellTypeStr = Const::CellTypeStrings[i];
        float posY = toFloat(i * rowHeight) + 2.0f;
        float posX = 5.0f;

        // Render background to glyph area
        for (int py = toInt(posY); py < toInt(posY) + rowHeight - 2; ++py) {
            for (int px = 3; px < toInt(textSizes.at(i).x) + 7; ++px) {
                int idx = (py * textureWidth + px) * 4;
                pixels[idx + 0] = 255;   // R
                pixels[idx + 1] = 255;   // G
                pixels[idx + 2] = 255;   // B
                pixels[idx + 3] = 20;  // A
            }
        }
        for (int py = toInt(posY) + 1; py < toInt(posY) + rowHeight - 4; ++py) {
            for (int px = 4; px < toInt(textSizes.at(i).x) + 6; ++px) {
                int idx = (py * textureWidth + px) * 4;
                pixels[idx + 0] = 0;  // R
                pixels[idx + 1] = 0;  // G
                pixels[idx + 2] = 0;  // B
                pixels[idx + 3] = 20;  // A
            }
        }

        // Render each character
        for (size_t charIdx = 0; charIdx < cellTypeStr.length(); ++charIdx) {
            char character = cellTypeStr[charIdx];
            auto glyph = font->FindGlyph((ImWchar)character);
            CHECK(glyph);

            // Calculate glyph position and size
            float x0 = posX + glyph->X0 * scale;
            float y0 = posY + glyph->Y0 * scale;
            float x1 = posX + glyph->X1 * scale;
            float y1 = posY + glyph->Y1 * scale;

            // Get texture coordinates in font atlas
            float u0 = glyph->U0;
            float v0 = glyph->V0;
            float u1 = glyph->U1;
            float v1 = glyph->V1;

            // Render glyph to our texture buffer
            for (int py = toInt(y0); py <= toInt(y1); ++py) {
                for (int px = toInt(x0); px <= toInt(x1); ++px) {
                    // Calculate texture coordinate in font atlas
                    float tu = u0 + (u1 - u0) * ((px - x0) / (x1 - x0));
                    float tv = v0 + (v1 - v0) * ((py - y0) / (y1 - y0));

                    int atlasx = toInt(tu * atlasWidth);
                    int atlasy = toInt(tv * atlasHeight);

                    int idx = (py * textureWidth + px) * 4;

                    if (atlasx >= 0 && atlasx < atlasWidth && atlasy >= 0 && atlasy < atlasHeight) {
                        auto alpha = atlasData[atlasy * atlasWidth + atlasx];

                        if (alpha > 0) {
                            // Write to our texture buffer (white text with alpha from font)
                            pixels[idx + 0] = 255;    // R
                            pixels[idx + 1] = 255;    // G
                            pixels[idx + 2] = 255;    // B
                            pixels[idx + 3] = toInt(toFloat(alpha) * 0.5f);  // A
                        }
                    }
                }
            }

            // Advance position for next character
            posX += glyph->AdvanceX * scale;
        }
    }

    // Create OpenGL texture
    glGenTextures(1, &_cellTypeTextureAtlas);
    glBindTexture(GL_TEXTURE_2D, _cellTypeTextureAtlas);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureWidth, textureHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

    glBindTexture(GL_TEXTURE_2D, 0);
}

SelectedConnectionRenderStep _SelectedConnectionRenderStep::create(StepParameters const& parameters)
{
    return SelectedConnectionRenderStep(new _SelectedConnectionRenderStep(parameters));
}

void _SelectedConnectionRenderStep::execute(ExecutionParameters parameters)
{
    auto zoom = Viewport::get().getZoomFactor();
    if (zoom <= ZoomFactorForCellDetails) {
        return;
    }

    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // Enable blending for arrows
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Draw connection arrows (geometry shader will convert to lines with arrows)
    glBindVertexArray(parameters._geometryBuffers->getVaoForSelectedConnections());
    glDrawArrays(GL_LINES, 0, toInt(parameters._geometryBuffers->getNumObjects().connectionArrowVertices));

    // Disable blending
    glDisable(GL_BLEND);
}

_SelectedConnectionRenderStep::_SelectedConnectionRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}

AttackEventRenderStep _AttackEventRenderStep::create(StepParameters const& parameters)
{
    return AttackEventRenderStep(new _AttackEventRenderStep(parameters));
}

void _AttackEventRenderStep::execute(ExecutionParameters parameters)
{
    if (!parameters._simulationParameters->attackVisualization.value) {
        return;
    }
    if (!_previousTargetSelection.has_value()) {
        parameters._clearBackground = true;
    }
    prepareExecution(parameters);

    // Enable blending for dashed lines
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);

    // Draw attack event lines (geometry shader will convert to dashed quads)
    glBindVertexArray(parameters._geometryBuffers->getVaoForAttackEvents());
    glDrawArrays(GL_LINES, 0, toInt(parameters._geometryBuffers->getNumObjects().attackEventVertices));

    // Disable blending
    glDisable(GL_BLEND);
}

_AttackEventRenderStep::_AttackEventRenderStep(StepParameters const& parameters)
    : _RenderStep(parameters)
{}
