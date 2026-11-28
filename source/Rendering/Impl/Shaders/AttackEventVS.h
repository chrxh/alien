#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const AttackEventVS = R"(
#version 450
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aIntensity;
layout (location = 3) in float aSeed;

out vec3 vertexColor;
out float vertexIntensity;
out float vertexSeed;

uniform vec2 rectUpperLeft;
uniform float zoom;

void main()
{
    // The geometry shader transforms the screen position to normalized device coordinates
    gl_Position = vec4((aPos - rectUpperLeft) * zoom, 0.0, 1.0);
    vertexColor = aColor;
    vertexIntensity = aIntensity;
    vertexSeed = aSeed;
}
)";
}
