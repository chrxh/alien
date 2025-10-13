#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 vertexColor[];
in float vertexZPos[];
out vec3 fragColor;

void main()
{
    // Create 3D positions for lighting calculation
    vec3 pos0 = vec3(gl_in[0].gl_Position.xy, vertexZPos[0]);
    vec3 pos1 = vec3(gl_in[1].gl_Position.xy, vertexZPos[1]);
    vec3 pos2 = vec3(gl_in[2].gl_Position.xy, vertexZPos[2]);
    
    // Calculate triangle normal using cross product
    vec3 edge1 = pos1 - pos0;
    vec3 edge2 = pos2 - pos0;
    vec3 normal = normalize(cross(edge1, edge2));
    
    // Light direction from front (camera looks at -Z, so light from +Z)
    vec3 lightDir = vec3(0.0, 0.0, 1.0);
    
    // Calculate lighting (dot product of normal and light direction)
    // Clamp to [0, 1] range
    float lightIntensity = max(0.0, dot(normal, lightDir));
    
    // Output the triangle with per-vertex lit colors (will be interpolated by GPU)
    for (int i = 0; i < 3; i++)
    {
        gl_Position = gl_in[i].gl_Position;
        // Apply lighting to each vertex color individually
        fragColor = mix(vertexColor[i], vec3(1.0, 1.0, 1.0), lightIntensity * 0.2);
        EmitVertex();
    }
    EndPrimitive();
}
