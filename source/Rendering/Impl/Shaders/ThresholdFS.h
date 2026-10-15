#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const ThresholdFS = R"(
#version 450
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float strength;

void main()
{
    // The glow takes its light from the bright regions, see SimulationRenderer.cpp. A soft threshold keeps the brightness continuous there.
    vec3 color = texture(inputTexture1, texCoord).rgb;
    float brightness = dot(color, vec3(1.0));
    FragColor = vec4(color * smoothstep(0.08, 0.28, brightness), 1.0);
}
)";
}
