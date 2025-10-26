#pragma once

#include <string_view>

namespace Shaders
{
    constexpr std::string_view CellTypeOverlayGS = R"(#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vColor[];
in int vCellType[];
in vec2 vWorldPos[];

out vec3 gColor;
out vec2 gQuadCoord;
out vec2 gTexCoord;
out vec2 gWorldPos;

uniform vec2 viewportSize;
uniform float zoom;
uniform float radius;

void main()
{
    int cellType = vCellType[0];
    
    gColor = vColor[0];
    gWorldPos = vWorldPos[0];
    
    // Calculate size in NDC coordinates
    float clampedZoom = min(40, zoom);
    float ndcWidth = 480.0 / viewportSize.x * 2.0 * clampedZoom / 30;
    float ndcHeight = 20.0 / viewportSize.y * 2.0 * clampedZoom / 30;
    
    // Get center position (cell position)
    vec4 center = gl_in[0].gl_Position;
    
    // Offset for text position (to the right of the cell)
    center.x = center.x + (radius * 0.25) / viewportSize.x;
    center.y = center.y - (radius * 0.25) / viewportSize.y;
    
    // Calculate texture coordinates for this cell type
    // Texture atlas has 13 rows (one per cell type), each row is 1/13 of texture height
    float texRowHeight = 20.0 / 512.0;
    float texMinY = float(cellType) * texRowHeight;
    float texMaxY = float(cellType + 1) * texRowHeight - 1.0 / 512.0;
    
    // Generate quad (4 vertices as triangle strip) positioned to the right of the cell
    // Bottom-left
    gl_Position = vec4(center.x, center.y - ndcHeight * 0.5, center.z, 1.0);
    gQuadCoord = vec2(0.0, 1.0);
    gTexCoord = vec2(0.0, texMaxY);
    EmitVertex();
    
    // Bottom-right
    gl_Position = vec4(center.x + ndcWidth, center.y - ndcHeight * 0.5, center.z, 1.0);
    gQuadCoord = vec2(1.0, 1.0);
    gTexCoord = vec2(1.0, texMaxY);
    EmitVertex();
    
    // Top-left
    gl_Position = vec4(center.x, center.y + ndcHeight * 0.5, center.z, 1.0);
    gQuadCoord = vec2(0.0, 0.0);
    gTexCoord = vec2(0.0, texMinY);
    EmitVertex();
    
    // Top-right
    gl_Position = vec4(center.x + ndcWidth, center.y + ndcHeight * 0.5, center.z, 1.0);
    gQuadCoord = vec2(1.0, 0.0);
    gTexCoord = vec2(1.0, texMinY);
    EmitVertex();
    
    EndPrimitive();
}
)";
}
