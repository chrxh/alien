#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const ModuloCopyVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 texCoord;
uniform vec2 viewportSize;
uniform vec2 worldSize;
uniform float zoom;

void main()
{
    //vec2 screenPos = aPos * zoom;
    //vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    //ndc.y = -ndc.y; // Flip Y coordinate
    
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);


    //gl_Position = vec4(aPos, 1.0);
    //vec2 temp = aTexCoord * clamp(worldSize * zoom / viewportSize, 0.0, 1.0);
    texCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
)";
}
