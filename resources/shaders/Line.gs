#version 330 core
layout (lines) in;
layout (line_strip, max_vertices = 2) out;

in vec3 vertexColor[];
in float vertexZPos[];
out vec3 fragColor;

#define PI 3.1415926538

void main()
{
    // Create 3D positions for lighting calculation
    vec3 pos0 = vec3(gl_in[0].gl_Position.xy, vertexZPos[0]);
    vec3 pos1 = vec3(gl_in[1].gl_Position.xy, vertexZPos[1]);
    
    // Calculate line direction and a perpendicular normal
    vec3 lineDir = normalize(pos1 - pos0);
    // For a 2D line in 3D space, create a normal perpendicular to the line
    // We'll use a normal that points "up" from the line in the XY plane
    vec3 normal = normalize(vec3(-lineDir.y, lineDir.x, 0.0));
    
    // Light direction (same as triangles)
    vec3 lightDir = normalize(vec3(1.0, 1.0, -1.0));
    
    // Calculate lighting (dot product of normal and light direction)
    // Clamp to [0, 1] range
    float lightIntensity = max(0.0, dot(normal, lightDir));
    
    // Output the line with per-vertex lit colors (will be interpolated by GPU)
    for (int i = 0; i < 2; i++)
    {
        gl_Position = gl_in[i].gl_Position;
        // Apply lighting to each vertex color individually (same formula as triangles)
        fragColor = vertexColor[i] * (0.8 + lightIntensity * 0.2);
        EmitVertex();
    }
    EndPrimitive();
}
