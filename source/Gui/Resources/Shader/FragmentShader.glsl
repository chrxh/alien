#version 330 core
out vec4 FragColor;
in vec2 texCoord;
uniform sampler2D texture1;
uniform int phase;

uniform float weight[10] = float[](0.9, 0.12, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01);

void main()
{
    vec2 texelSize = 1.0 / textureSize(texture1, 0);
    if (phase == 0) {
        vec2 mirroredCoord = vec2(texCoord.x, 1.0 - texCoord.y);  //mirror

        vec3 result = texture(texture1, mirroredCoord).rgb * weight[0];
        for (int i = 1; i < 10; ++i) {
            result += texture(texture1, mirroredCoord + vec2(texelSize.x * i, 0.0)).rgb * weight[i];
            result += texture(texture1, mirroredCoord - vec2(texelSize.x * i, 0.0)).rgb * weight[i];
        }
        FragColor = vec4(result, 1.0);
    } else {
        vec3 result = texture(texture1, texCoord).rgb * weight[0];
        for (int i = 1; i < 10; ++i) {
            result += texture(texture1, texCoord + vec2(0.0, texelSize.y * i)).rgb * weight[i];
            result += texture(texture1, texCoord - vec2(0.0, texelSize.y * i)).rgb * weight[i];
        }
        FragColor = vec4(result, 1.0);
    }
}