#include "MacroView.h"

#include <glad/glad.h>

#include "EngineImpl/SimulationController.h"

#include "Shader.h"

void MacroView::init(SimulationController* simController, IntVector2D const& viewportSize)
{
    _simController = simController;
    _shader =
        new Shader("d:\\temp\\alien-imgui\\source\\Gui\\texture.vs", "d:\\temp\\alien-imgui\\source\\Gui\\texture.fs");

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
        1.0f,  1.0f,  0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,  // top right
        1.0f,  -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,  // bottom right
        -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom left
        -1.0f, 1.0f,  0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f   // top left
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
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    resize(viewportSize);
}

void MacroView::resize(IntVector2D const& size)
{
    glGenTextures(1, &_textureId);
    glBindTexture(GL_TEXTURE_2D, _textureId);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16, size.x, size.y, 0, GL_RGB, GL_UNSIGNED_SHORT, NULL);

    _cudaResource = _simController->registerImageResource(_textureId);

    _shader->use();
    _shader->setInt("texture1", 0);
    _viewportSize = size;
}

void MacroView::render()
{
    _simController->getVectorImage({-50, 0}, {99, 99}, _cudaResource, {_viewportSize.x, _viewportSize.y}, 1);

    _shader->use();
    glBindVertexArray(_vao);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}
