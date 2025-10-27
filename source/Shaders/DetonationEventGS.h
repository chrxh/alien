#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DetonationEventGS = R"(
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in float vertexRadius[];
out vec2 quadCoord;  // Coordinates within the quad, from -1 to 1
out float fragRadius;

uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    // Get the center position in NDC
    vec4 center = gl_in[0].gl_Position;
    float radius = vertexRadius[0];
    
    // Pass radius to fragment shader
    fragRadius = radius;
    
    // Calculate size in NDC coordinates
    // The quad should be large enough to contain the circle
    float ndcHalfWidth = (radius * zoom) / viewportSize.x * 2.0;
    float ndcHalfHeight = (radius * zoom) / viewportSize.y * 2.0;
    
    // Generate quad (4 vertices as triangle strip)
    // Bottom-left
    gl_Position = vec4(center.xy + vec2(-ndcHalfWidth, -ndcHalfHeight), center.z, 1.0);
    quadCoord = vec2(-1.0, -1.0);
    EmitVertex();
    
    // Bottom-right
    gl_Position = vec4(center.xy + vec2(ndcHalfWidth, -ndcHalfHeight), center.z, 1.0);
    quadCoord = vec2(1.0, -1.0);
    EmitVertex();
    
    // Top-left
    gl_Position = vec4(center.xy + vec2(-ndcHalfWidth, ndcHalfHeight), center.z, 1.0);
    quadCoord = vec2(-1.0, 1.0);
    EmitVertex();
    
    // Top-right
    gl_Position = vec4(center.xy + vec2(ndcHalfWidth, ndcHalfHeight), center.z, 1.0);
    quadCoord = vec2(1.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
)";
}
