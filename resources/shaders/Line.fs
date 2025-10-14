#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main()
{
    // Use the color from vertex shader (includes lighting)
    FragColor = vec4(fragColor, 1.0);
}
