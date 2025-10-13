#version 330 core
layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in vec3 vertexColor[];
out vec3 fragColor;

void main()
{
    // Calculate average color of the three vertices
    vec3 avgColor = (vertexColor[0] + vertexColor[1] + vertexColor[2]) / 3.0;
    
    // Output the triangle with the averaged color
    for (int i = 0; i < 3; i++)
    {
        gl_Position = gl_in[i].gl_Position;
        fragColor = avgColor;
        EmitVertex();
    }
    EndPrimitive();
}
