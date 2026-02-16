#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedConnectionGS = R"(
#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 20) out;

in vec3 vertexColor[];
flat in float weightToObject1[];
flat in float weightToObject2[];
out vec3 fragColor;

uniform vec2 viewportSize;
uniform float zoom;

#define PI 3.1415926538

vec2 transform(vec2 v)
{
    v = v / viewportSize * 2.0 - 1.0;
    return vec2(v.x, -v.y);
}

void emitLine(vec4 p0, vec4 p1, vec3 color0, vec3 color1, float lineWidth)
{
    // Calculate line direction in screen space
    vec2 dir = normalize(p1.xy - p0.xy);
    
    // Calculate perpendicular direction (for line thickness)
    vec2 perp = vec2(-dir.y, dir.x);
    
    vec2 offset = perp * lineWidth * 0.5;
    
    // Generate quad (4 vertices as triangle strip)
    gl_Position = vec4(transform(p0.xy - offset), 0.0, 1.0);
    fragColor = color0;
    EmitVertex();
    
    gl_Position = vec4(transform(p0.xy + offset), 0.0, 1.0);
    fragColor = color0;
    EmitVertex();
    
    gl_Position = vec4(transform(p1.xy - offset), 0.0, 1.0);
    fragColor = color1;
    EmitVertex();
    
    gl_Position = vec4(transform(p1.xy + offset), 0.0, 1.0);
    fragColor = color1;
    EmitVertex();
    
    EndPrimitive();
}

void emitArrowHead(vec4 basePos, vec2 dir, vec3 color, float lineWidth, float arrowSize)
{
    // Create arrowhead at the end of the line
    // Arrow has 90-degree tip with both sides at 45 degrees to the baseline
    
    // Rotate by -45 degrees (clockwise) and reverse: both sides point backward from tip
    vec2 arrowDir1 = normalize(vec2(-dir.x - dir.y, dir.x - dir.y)) * arrowSize;
    // Rotate by +45 degrees (counter-clockwise) and reverse
    vec2 arrowDir2 = normalize(vec2(-dir.x + dir.y, -dir.x - dir.y)) * arrowSize;
    
    vec4 arrowTip = basePos;
    vec4 arrowPoint1 = vec4(basePos.xy + arrowDir1, 0.0, 1.0);
    vec4 arrowPoint2 = vec4(basePos.xy + arrowDir2, 0.0, 1.0);
    
    // First arrow line
    vec2 perp1 = normalize(vec2(-arrowDir1.y, arrowDir1.x));
    vec2 offset1 = perp1 * lineWidth * 0.4;
    
    gl_Position = vec4(transform(arrowTip.xy - offset1), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    gl_Position = vec4(transform(arrowTip.xy + offset1), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    gl_Position = vec4(transform(arrowPoint1.xy - offset1), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    gl_Position = vec4(transform(arrowPoint1.xy + offset1), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    EndPrimitive();
    
    // Second arrow line
    vec2 perp2 = normalize(vec2(-arrowDir2.y, arrowDir2.x));
    vec2 offset2 = perp2 * lineWidth * 0.4;
    
    gl_Position = vec4(transform(arrowTip.xy - offset2), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    gl_Position = vec4(transform(arrowTip.xy + offset2), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    gl_Position = vec4(transform(arrowPoint2.xy - offset2), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    gl_Position = vec4(transform(arrowPoint2.xy + offset2), 0.0, 1.0);
    fragColor = color;
    EmitVertex();
    
    EndPrimitive();
}

void main()
{
    // Get the two vertices of the line in NDC
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;

    // Get connection weights
    float cwToObject1 = weightToObject1[0];
    float cwToObject2 = weightToObject2[0];
    
    // Line width in NDC coordinates - thinner lines (1-2 pixels)
    float lineWidth = 2.0;
    
    // Calculate line direction
    vec2 dir = normalize(p1.xy - p0.xy);
	p0.xy = p0.xy + dir * 0.28 * zoom;
	p1.xy = p1.xy - dir * 0.28 * zoom;
    
    // Draw the main line
    emitLine(p0, p1, vertexColor[0], vertexColor[1], lineWidth);
    
    // Draw arrow heads scaled by connection weight (analogous to CreaturePreviewWidget)
    if (cwToObject1 != 0.0) {
        float arrowScale = min(abs(cwToObject1), 1.0);
        float arrowSize = zoom / 8.0 * arrowScale;
        emitArrowHead(p0, -dir, vertexColor[0], lineWidth, arrowSize);
    }
    
    if (cwToObject2 != 0.0) {
        float arrowScale = min(abs(cwToObject2), 1.0);
        float arrowSize = zoom / 8.0 * arrowScale;
        emitArrowHead(p1, dir, vertexColor[1], lineWidth, arrowSize);
    }
}
)";
}
