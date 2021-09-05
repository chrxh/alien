#version 330 core
out vec4 FragColor;

in vec3 ourColor;
in vec2 TexCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform int phase;
uniform bool glowEffect;
uniform bool motionEffect;
uniform float motionBlurFactor;

uniform float weight_rg[10] = float[](0.7, 0.16, 0.03, 0.03, 0.03, 0.03, 0.01, 0.01, 0.01, 0.01);
uniform float weight_b[10] = float[](0.5, 0.12, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02);

void main()
{
    vec2 texelSize = 1.0 / textureSize(texture1, 0);
    if (phase == 0) {
        vec2 mirroredCoord = vec2(TexCoord.x, 1.0 - TexCoord.y);  //mirror

        vec3 result;
        if (glowEffect) {
            result = vec3(texture(texture1, mirroredCoord).rg * weight_rg[0], texture(texture1, mirroredCoord).b * weight_b[0]);
            for (int i = 1; i < 10; ++i) {
                result.rg += texture(texture1, mirroredCoord + vec2(texelSize.x * i, 0.0)).rg * weight_rg[i];
                result.rg += texture(texture1, mirroredCoord - vec2(texelSize.x * i, 0.0)).rg * weight_rg[i];

                result.b += texture(texture1, mirroredCoord + vec2(texelSize.x * i, 0.0)).b * weight_b[i];
                result.b += texture(texture1, mirroredCoord - vec2(texelSize.x * i, 0.0)).b * weight_b[i];
                result.b += texture(texture1, mirroredCoord + vec2(0.0, texelSize.y * i)).b * weight_b[i];
                result.b += texture(texture1, mirroredCoord - vec2(0.0, texelSize.y * i)).b * weight_b[i];
            }
        } else {
            result = vec3(texture(texture1, mirroredCoord).rgb);
        }
        result = sqrt(result * 250.0) - 0.2;
        if (motionEffect) {
            result = result * motionBlurFactor + texture(texture2, TexCoord).rgb * (1 - motionBlurFactor);
        }
        FragColor = vec4(result, 1.0);
    } else {
        vec3 result;
        if (glowEffect) {
            result =
                vec3(texture(texture1, TexCoord).rg * weight_rg[0], texture(texture1, TexCoord).b * weight_b[0]);
            for (int i = 1; i < 10; ++i) {
                result.rg += texture(texture1, TexCoord + vec2(0.0, texelSize.y * i)).rg * weight_rg[i];
                result.rg += texture(texture1, TexCoord - vec2(0.0, texelSize.y * i)).rg * weight_rg[i];

                result.b += texture(texture1, TexCoord + vec2(0.0, texelSize.y * i)).b * weight_b[i];
                result.b += texture(texture1, TexCoord - vec2(0.0, texelSize.y * i)).b * weight_b[i];
                result.b += texture(texture1, TexCoord + vec2(texelSize.x * i, 0.0)).b * weight_b[i];
                result.b += texture(texture1, TexCoord - vec2(texelSize.x * i, 0.0)).b * weight_b[i];
            }
        } else {
            result = vec3(texture(texture1, TexCoord).rgb);
        }
        FragColor = vec4(result, 1.0);
    }
}
