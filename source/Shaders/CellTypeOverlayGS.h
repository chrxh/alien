#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const CellTypeOverlayGS = R"(
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vColor[];
in int vCellType[];
in int vObjectType[];
in vec2 vWorldPos[];

out vec3 gColor;
out vec2 gQuadCoord;
out vec2 gTexCoord;
out vec2 gWorldPos;

uniform vec2 viewportSize;
uniform float zoom;
uniform float radius;

// Object type constants (matching ObjectType_ enum)
const int ObjectType_Structure = 0;
const int ObjectType_Fluid = 1;
const int ObjectType_FreeCell = 2;
const int ObjectType_Cell = 3;

// Number of cell types (matching CellType_Count)
const int CellType_Count = 15;

void main()
{
    int objectType = vObjectType[0];
    int cellType = vCellType[0];
    
    // Determine texture row based on object type:
    // - ObjectType_Cell: use cell type (rows 0-12)
    // - ObjectType_Structure: row 13 ("Structure")
    // - ObjectType_Fluid: row 14 ("Fluid")
    // - ObjectType_FreeCell: row 15 ("Free Cell")
    int textureRow;
    if (objectType == ObjectType_Cell) {
        textureRow = cellType;
    } else if (objectType == ObjectType_Structure) {
        textureRow = CellType_Count;  // Row 13
    } else if (objectType == ObjectType_Fluid) {
        textureRow = CellType_Count + 1;  // Row 14
    } else {
        textureRow = CellType_Count + 2;  // Row 15 (ObjectType_FreeCell)
    }
    
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
    
    // Calculate texture coordinates for this row
    // Texture atlas has 16 rows (13 cell types + 3 object types), each row is 20 pixels of texture height
    float texRowHeight = 20.0 / 512.0;
    float texMinY = float(textureRow) * texRowHeight;
    float texMaxY = float(textureRow + 1) * texRowHeight - 1.0 / 512.0;
    
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
