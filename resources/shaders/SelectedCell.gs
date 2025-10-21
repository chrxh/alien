#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec2 vWorldPos[];

out vec2 gWorldPos;
out vec2 gQuadCoord;

uniform vec2 viewportSize;
uniform float zoom;
uniform float radius;

void main()
{
    gWorldPos = vWorldPos[0];
    
    // Circle radius in world coordinates (thin white circle)
    float circleRadius = radius * 0.3;  // Half the cell size for visibility
    
    // Calculate size in NDC coordinates
    float ndcHalfWidth = circleRadius / viewportSize.x * 2.0;
    float ndcHalfHeight = circleRadius / viewportSize.y * 2.0;
    
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
