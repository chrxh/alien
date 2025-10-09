#include "RenderStep.h"

#include <EngineInterface/RenderData.h>

#include "Shader.h"

_RenderStep::_RenderStep(std::filesystem::path const& vertexShader, std::filesystem::path const& fragmentShader, std::vector<RenderStep> const& dependentSteps)
    : _dependentSteps(dependentSteps)
{
    _shader = std::make_shared<_Shader>(vertexShader, fragmentShader);
}

_PointRenderStep::_PointRenderStep(
    std::filesystem::path const& vertexShader,
    std::filesystem::path const& fragmentShader,
    std::vector<RenderStep> const& dependentSteps)
    : _RenderStep(vertexShader, fragmentShader, dependentSteps)
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

void _PostProcessingRenderStep::execute() {}
