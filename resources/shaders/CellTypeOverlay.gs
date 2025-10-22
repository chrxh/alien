#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vColor[];
in float vState[];
in vec2 vWorldPos[];

out vec3 gColor;
out vec2 gQuadCoord;
out vec2 gWorldPos;

uniform vec2 viewportSize;
uniform float zoom;
uniform float radius;

void main()
{
    gColor = vColor[0];
    gWorldPos = vWorldPos[0];
    
    // Size of the text quad in world coordinates
    // Make it proportional to zoom but with a minimum size
    float quadWidth = max(2.0, 1.5 * radius);
    float quadHeight = max(1.0, 0.5 * radius);
    
    // Position offset: place text to the right of the cell
    float offsetX = radius * 0.8;
    float offsetY = 0.0;
    
    // Calculate size in NDC coordinates
    float ndcHalfWidth = quadWidth / viewportSize.x;
    float ndcHalfHeight = quadHeight / viewportSize.y;
    float ndcOffsetX = offsetX / viewportSize.x * 2.0;
    float ndcOffsetY = offsetY / viewportSize.y * 2.0;
    
    // Get center position
    vec4 center = gl_in[0].gl_Position;
    vec2 offset = vec2(ndcOffsetX, ndcOffsetY);
    
    // Generate quad (4 vertices as triangle strip)
    // Bottom-left
    gl_Position = vec4(center.xy + offset + vec2(-ndcHalfWidth, -ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(0.0, 1.0);
    EmitVertex();
    
    // Bottom-right
    gl_Position = vec4(center.xy + offset + vec2(ndcHalfWidth, -ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(1.0, 1.0);
    EmitVertex();
    
    // Top-left
    gl_Position = vec4(center.xy + offset + vec2(-ndcHalfWidth, ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    // Top-right
    gl_Position = vec4(center.xy + offset + vec2(ndcHalfWidth, ndcHalfHeight), center.z, 1.0);
    gQuadCoord = vec2(1.0, 0.0);
    EmitVertex();
    
    EndPrimitive();
}
