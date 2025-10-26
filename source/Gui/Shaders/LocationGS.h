#pragma once

#include <string_view>

namespace Shaders
{
    constexpr std::string_view LocationGS = R"(#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vColor[];
in vec2 vWorldPos[];
flat in int vShapeType[];
in float vDimension1[];
in float vDimension2[];
in float vFadeoutRadius[];
in float vOpacity[];

out vec3 gColor;
out vec2 gWorldPos;
flat out int gShapeType;
out float gDimension1;
out float gDimension2;
out float gFadeoutRadius;
out float gOpacity;
out vec2 gQuadCoord;  // Coordinates within the quad, from -0.5 to 0.5

uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    // Pass through all attributes
    gColor = vColor[0];
    gWorldPos = vWorldPos[0];
    gShapeType = vShapeType[0];
    gDimension1 = vDimension1[0];
    gDimension2 = vDimension2[0];
    gFadeoutRadius = vFadeoutRadius[0];
    gOpacity = vOpacity[0];
    
    // Calculate the size of the quad in world coordinates
    float maxDim;
    if (vShapeType[0] == 0) {
        // Circular: diameter + fadeout on both sides
        maxDim = (vDimension1[0] + vFadeoutRadius[0]) * 2.0;
    } else {
        // Rectangular: max of width and height + fadeout
        maxDim = max(vDimension1[0] + vFadeoutRadius[0] * 2, vDimension2[0] + vFadeoutRadius[0] * 2);
    }
    
    // Add some padding for anti-aliasing (4 pixels worth in world space)
    float padding = 4.0 / zoom;
    float halfSize = maxDim * 0.5 + padding;
    
    // Calculate size in NDC coordinates
    float ndcHalfWidth = (halfSize * zoom) / viewportSize.x * 2.0;
    float ndcHalfHeight = (halfSize * zoom) / viewportSize.y * 2.0;
    
    // Get center position
    vec4 center = gl_in[0].gl_Position;
    
    // Generate quad (4 vertices as triangle strip)
    // Bottom-left
    gl_Position = vec4(center.xy + vec2(-ndcHalfWidth, -ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(-0.5, 0.5);
    EmitVertex();
    
    // Bottom-right
    gl_Position = vec4(center.xy + vec2(ndcHalfWidth, -ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(0.5, 0.5);
    EmitVertex();
    
    // Top-left
    gl_Position = vec4(center.xy + vec2(-ndcHalfWidth, ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(-0.5, -0.5);
    EmitVertex();
    
    // Top-right
    gl_Position = vec4(center.xy + vec2(ndcHalfWidth, ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(0.5, -0.5);
    EmitVertex();
    
    EndPrimitive();
}
)";
}
