#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineFS = R"(
#version 330 core
in vec3 fragColor;
in float edgeDist;
in float coreEdge;
out vec4 FragColor;

void main()
{
    // Core region at full brightness, fade only in the AA margin
    float edgeAlpha = 1.0 - smoothstep(coreEdge, 1.0, abs(edgeDist));

    // Push edge pixels to far depth so triangles can overwrite them;
    // core pixels (edgeAlpha ~1.0) keep normal depth to block overwrites
    if (edgeAlpha < 0.99) {
        gl_FragDepth = 1.0;
    } else {
        gl_FragDepth = gl_FragCoord.z;
    }

    FragColor = vec4(fragColor, edgeAlpha);
}
)";
}
