#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const LineFS = R"(
#version 330 core
in vec3 fragColor;
in vec2 lineCoord;
flat in float lineLength;
flat in float coreHalfWidth;
flat in float aaMargin;
out vec4 FragColor;

void main()
{
    float capDistance = max(max(-lineCoord.x, lineCoord.x - lineLength), 0.0);
    float distanceToLine = length(vec2(capDistance, lineCoord.y));
    float edgeAlpha = 1.0 - smoothstep(coreHalfWidth, coreHalfWidth + aaMargin, distanceToLine);

    if (edgeAlpha <= 0.0) {
        discard;
    }

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
