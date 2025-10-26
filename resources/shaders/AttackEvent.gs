#version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices = 4) out;

in vec3 vertexColor[];
out vec3 fragColor;
out vec2 lineCoord;

uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    // Get the two vertices of the line in NDC
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    
    // Calculate line direction in screen space
    vec2 dir = normalize(p1.xy - p0.xy);
    
    // Calculate perpendicular direction (for line thickness)
    vec2 perp = vec2(-dir.y, dir.x);
    
    // Line width in NDC coordinates (same as Line.gs)
    float lineWidth = (zoom * 0.15) / viewportSize.x * 2.0;
    vec2 offset = perp * lineWidth * 0.5;
    
    // Calculate line length for dashed pattern
    float lineLength = length(p1.xy - p0.xy);
    
    // Use same color for both vertices (red for attacked)
    vec3 color = vertexColor[0];
    
    // Generate quad (4 vertices as triangle strip)
    // Vertex 0 (bottom-left)
    gl_Position = vec4(p0.xy - offset, 0.0, 1.0);
    fragColor = color;
    lineCoord = vec2(0.0, 0.0);
    EmitVertex();
    
    // Vertex 1 (top-left)
    gl_Position = vec4(p0.xy + offset, 0.0, 1.0);
    fragColor = color;
    lineCoord = vec2(0.0, 1.0);
    EmitVertex();
    
    // Vertex 2 (bottom-right)
    gl_Position = vec4(p1.xy - offset, 0.0, 1.0);
    fragColor = color;
    lineCoord = vec2(lineLength, 0.0);
    EmitVertex();
    
    // Vertex 3 (top-right)
    gl_Position = vec4(p1.xy + offset, 0.0, 1.0);
    fragColor = color;
    lineCoord = vec2(lineLength, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
