#include "MacroView.h"

#include <glad/glad.h>

#include "EngineImpl/SimulationController.h"

#include "Shader.h"

void _MacroView::init(SimulationController const& simController, IntVector2D const& viewportSize, float zoomFactor)
{
    auto worldSize = simController->getWorldSize();
    _worldCenter = {toFloat(worldSize.x) / 2, toFloat(worldSize.y) / 2};
    _zoomFactor = zoomFactor;

    _simController = simController;
    _shader = boost::make_shared<_Shader>("d:\\temp\\alien-imgui\\source\\Gui\\texture.vs", "d:\\temp\\alien-imgui\\source\\Gui\\texture.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions        // texture coordinates
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
    glGenVertexArrays(1, &_vao);
    glGenBuffers(1, &_vbo);
    glGenBuffers(1, &_ebo);

    glBindVertexArray(_vao);

    glBindBuffer(GL_ARRAY_BUFFER, _vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
  
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    resize(viewportSize);

    _shader->use();
    _shader->setInt("texture1", 0);
    _shader->setInt("texture2", 0);
    _shader->setBool("glowEffect", true);
    _shader->setBool("motionEffect", true);
    _shader->setFloat("motionBlurFactor", 0.6f);
}

void _MacroView::resize(IntVector2D const& size)
{
    if (_areTexturesInitialized) {
        glDeleteFramebuffers(1, &_fbo);
        glDeleteTextures(1, &_textureId);
        glDeleteTextures(1, &_textureFramebufferId);

        _areTexturesInitialized = true;
    }

    glGenTextures(1, &_textureId);
    glBindTexture(GL_TEXTURE_2D, _textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_SHORT, NULL);
    _cudaResource = _simController->registerImageResource(_textureId);

    glGenTextures(1, &_textureFramebufferId);
    glBindTexture(GL_TEXTURE_2D, _textureFramebufferId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_SHORT, NULL);

    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);  
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _textureFramebufferId, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);  

    _viewportSize = size;
}

void _MacroView::leftMouseButtonHold(IntVector2D const& viewPos)
{
    auto worldPos = mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
    _zoomFactor *= 1.02f;
    centerTo(worldPos, viewPos);
}

void _MacroView::rightMouseButtonHold(IntVector2D const& viewPos)
{
    auto worldPos = mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
    _zoomFactor /= 1.02f;
    centerTo(worldPos, viewPos);
}

void _MacroView::middleMouseButtonPressed(IntVector2D const& viewPos)
{
    _worldPosForMovement = mapViewToWorldPosition({toFloat(viewPos.x), toFloat(viewPos.y)});
}

void _MacroView::middleMouseButtonHold(IntVector2D const& viewPos)
{
    centerTo(*_worldPosForMovement, viewPos);
}

void _MacroView::middleMouseButtonReleased()
{
    _worldPosForMovement = boost::none;
}

void _MacroView::render()
{
    requestImageFromSimulation();

    _shader->use();

    GLint currentFbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &currentFbo);

    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
    _shader->setInt("phase", 0);
    glBindVertexArray(_vao);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _textureId);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, currentFbo);
    _shader->setInt("phase", 1);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _textureFramebufferId);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}

void _MacroView::requestImageFromSimulation()
{
    auto topLeft = mapViewToWorldPosition(RealVector2D{0, 0});
    auto bottomRight = mapViewToWorldPosition(RealVector2D{toFloat(_viewportSize.x - 1), toFloat(_viewportSize.y - 1)});

    _simController->getVectorImage(topLeft, bottomRight, _cudaResource, {_viewportSize.x, _viewportSize.y}, _zoomFactor);
}

void _MacroView::centerTo(RealVector2D const& worldPosition, IntVector2D const& viewPos)
{
    RealVector2D deltaViewPos{
        toFloat(viewPos.x) - toFloat(_viewportSize.x) / 2.0f, toFloat(viewPos.y) - toFloat(_viewportSize.y) / 2.0f};
    auto deltaWorldPos = deltaViewPos / _zoomFactor;
    _worldCenter = worldPosition - deltaWorldPos;
}

RealVector2D _MacroView::mapViewToWorldPosition(RealVector2D const& viewPos) const
{
    RealVector2D relCenter{
        toFloat(_viewportSize.x / (2.0 * _zoomFactor)), toFloat(_viewportSize.y / (2.0 * _zoomFactor))};
    RealVector2D relWorldPos{viewPos.x / _zoomFactor, viewPos.y / _zoomFactor};
    return _worldCenter - relCenter + relWorldPos;
}
