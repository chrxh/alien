#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const AttackEventVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform vec2 viewportSize;

void main()
{
    // Transform world position to screen coordinates
    // Note: Transform to normalized device coordinates is done in geometry shader.
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    
    gl_Position = vec4(screenPos, 0.0, 1.0);
    
    // Pass color to geometry shader
    vertexColor = aColor;
}
)";
}
