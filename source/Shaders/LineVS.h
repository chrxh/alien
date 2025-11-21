#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int isActive;

out vec3 vertexColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform vec2 viewportSize;

void main()
{
    // Transform world position to screen coordinates
    // Note: Transform to normalized device coordinates is done in geometry shader.
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    
    gl_Position = vec4(screenPos, aPos.z, 1.0);
    
    // Pass color to geometry shader
    vertexColor = aColor;
}
)";
}
