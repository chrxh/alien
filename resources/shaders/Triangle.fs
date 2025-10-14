#version 330 core
in vec3 fragColor;
out vec4 FragColor;

void main()
{
    // Use the averaged color from geometry shader with some transparency
    FragColor = vec4(fragColor, 0.3);
}
