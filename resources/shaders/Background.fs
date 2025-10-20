#version 330 core
out vec4 FragColor;

uniform vec2 viewportSize;
uniform float zoom;
uniform vec3 background;

void main()
{
    FragColor = vec4(background, 1.0);
}
