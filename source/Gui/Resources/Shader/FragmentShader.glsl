#version 330 core
out vec4 FragColor;
in vec2 texCoord;
uniform sampler2D texture1;
uniform bool mirror;
void main()
{
    if (mirror) {
        FragColor = texture(texture1, vec2(texCoord.x, 1.0 - texCoord.y));
    } else {
        FragColor = texture(texture1, texCoord);
    }
}