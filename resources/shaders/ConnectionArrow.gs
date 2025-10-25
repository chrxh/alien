#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 18) out;

in vec3 vertexColor[];
flat in int vertexActive[];
out vec3 fragColor;

uniform vec2 viewportSize;
uniform float zoom;

#define PI 3.1415926538

void emitLine(vec4 p0, vec4 p1, vec3 color0, vec3 color1, float lineWidth)
{
    // Calculate line direction in screen space
    vec2 dir = normalize(p1.xy - p0.xy);
    
    // Calculate perpendicular direction (for line thickness)
    vec2 perp = vec2(-dir.y, dir.x);
    
    vec2 offset = perp * lineWidth * 0.5;
    
    // Create 3D positions for lighting calculation
    vec3 pos0 = vec3(p0.xyz);
    vec3 pos1 = vec3(p1.xyz);
    
    // Calculate line direction and a perpendicular normal for lighting
    vec3 lineDir = normalize(pos1 - pos0);
    vec3 normal = normalize(vec3(-lineDir.y, lineDir.x, 0.0));
    
    // Light direction (same as triangles)
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
    
    // Calculate lighting (dot product of normal and light direction)
    float lightIntensity = max(0.0, dot(normal, lightDir));
    
    // Apply lighting to colors (same formula as triangles)
    vec3 litColor0 = color0 * (0.8 + lightIntensity * 0.2);
    vec3 litColor1 = color1 * (0.8 + lightIntensity * 0.2);
    
    // Generate quad (4 vertices as triangle strip)
    gl_Position = vec4(p0.xy - offset, p0.z, 1.0);
    fragColor = litColor0;
    EmitVertex();
    
    gl_Position = vec4(p0.xy + offset, p0.z, 1.0);
    fragColor = litColor0;
    EmitVertex();
    
    gl_Position = vec4(p1.xy - offset, p1.z, 1.0);
    fragColor = litColor1;
    EmitVertex();
    
    gl_Position = vec4(p1.xy + offset, p1.z, 1.0);
    fragColor = litColor1;
    EmitVertex();
    
    EndPrimitive();
}

void emitArrowHead(vec4 basePos, vec2 dir, vec3 color, float lineWidth, float arrowSize)
{
    // Create arrowhead at the end of the line
    // Two lines forming a V shape pointing in the direction
    
    vec3 pos = vec3(basePos.xyz);
    vec3 normal = normalize(vec3(-dir.y, dir.x, 0.0));
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
    float lightIntensity = max(0.0, dot(normal, lightDir));
    vec3 litColor = color * (0.8 + lightIntensity * 0.2);
    
    // Calculate arrow parts - they point backward from the end position
    vec2 arrowDir1 = normalize(vec2(-dir.x + dir.y, -dir.x - dir.y)) * arrowSize;
    vec2 arrowDir2 = normalize(vec2(-dir.x - dir.y, dir.x - dir.y)) * arrowSize;
    
    vec4 arrowTip = basePos;
    vec4 arrowPoint1 = vec4(basePos.xy + arrowDir1, basePos.z, 1.0);
    vec4 arrowPoint2 = vec4(basePos.xy + arrowDir2, basePos.z, 1.0);
    
    // First arrow line
    vec2 perp1 = normalize(vec2(-arrowDir1.y, arrowDir1.x));
    vec2 offset1 = perp1 * lineWidth * 0.4;
    
    gl_Position = vec4(arrowTip.xy - offset1, arrowTip.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    gl_Position = vec4(arrowTip.xy + offset1, arrowTip.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    gl_Position = vec4(arrowPoint1.xy - offset1, arrowPoint1.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    gl_Position = vec4(arrowPoint1.xy + offset1, arrowPoint1.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    EndPrimitive();
    
    // Second arrow line
    vec2 perp2 = normalize(vec2(-arrowDir2.y, arrowDir2.x));
    vec2 offset2 = perp2 * lineWidth * 0.4;
    
    gl_Position = vec4(arrowTip.xy - offset2, arrowTip.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    gl_Position = vec4(arrowTip.xy + offset2, arrowTip.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    gl_Position = vec4(arrowPoint2.xy - offset2, arrowPoint2.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    gl_Position = vec4(arrowPoint2.xy + offset2, arrowPoint2.z, 1.0);
    fragColor = litColor;
    EmitVertex();
    
    EndPrimitive();
}

void main()
{
    // Get the two vertices of the line in NDC
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    
    // Extract arrow direction flags from the state field
    // Bit 0: arrow to cell1 (vertex 0)
    // Bit 1: arrow to cell2 (vertex 1)
    bool arrowToCell1 = (vertexActive[0] & 1) != 0;
    bool arrowToCell2 = (vertexActive[0] & 2) != 0;
    
    // Line width in NDC coordinates - thinner lines (1-2 pixels)
    float lineWidth = (2.0) / viewportSize.x * 2.0;
    // Arrow size - make arrows more visible
    float arrowSize = zoom / 8.0 * 2.0 / viewportSize.x;
    
    // Average color for the connection
    vec3 avgColor = (vertexColor[0] + vertexColor[1]) * 0.5;
    
    // Calculate line direction
    vec2 dir = normalize(p1.xy - p0.xy);
    
    // Draw the main line
    emitLine(p0, p1, avgColor, avgColor, lineWidth);
    
    // Draw arrow heads if signal can flow
    if (arrowToCell1) {
        emitArrowHead(p0, -dir, avgColor, lineWidth, arrowSize);
    }
    
    if (arrowToCell2) {
        emitArrowHead(p1, dir, avgColor, lineWidth, arrowSize);
    }
}
