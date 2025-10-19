#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float strength;

void main()
{
    vec3 color = texture(inputTexture1, texCoord).rgb;
    float brightness = dot(color, vec3(1.0));
    FragColor = vec4(brightness > 0.3 ? color : vec3(0.0), 1.0);
}
