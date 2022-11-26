#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform int phase;
uniform bool glowEffect;
uniform bool motionEffect;
uniform float motionBlurFactor;
uniform float brightness;
uniform float contrast;

uniform float weights[10] = float[](0.22, 0.22, 0.12, 0.07, 0.05, 0.05, 0.04, 0.04, 0.03, 0.02);

vec3 mapColor(vec3 color)
{
    return (((sqrt(color * 256.0) - 0.2) - 0.5)* contrast + 0.5) * brightness;
}

void main()
{
    vec2 texelSize = 1.0 / textureSize(texture1, 0);
    vec2 mirroredCoord = vec2(texCoord.x, 1.0 - texCoord.y);

    //horizontal blur
    if (phase == 0) {
        vec3 result;
        if (glowEffect) {
            result = vec3(texture(texture1, mirroredCoord).rgb * weights[0]);
            for (int i = 1; i < 10; ++i) {
                result += texture(texture1, mirroredCoord + vec2(texelSize.x * i, 0.0)).rgb * weights[i];
                result += texture(texture1, mirroredCoord - vec2(texelSize.x * i, 0.0)).rgb * weights[i];
            }
        } else {
            result = vec3(texture(texture1, mirroredCoord).rgb);
        }
        result = mapColor(result);
        if (motionEffect) {
            result = result * motionBlurFactor + texture(texture2, texCoord).rgb * (1 - motionBlurFactor);
        }
        FragColor = vec4(result, 1.0);
    }

    //vertical blur
    if(phase == 1) {
        vec3 result;
        if (glowEffect) {
            result =
                vec3(texture(texture2, texCoord).rgb * weights[0]);
            for (int i = 1; i < 10; ++i) {
                result += texture(texture2, texCoord + vec2(0.0, texelSize.y * i)).rgb * weights[i];
                result += texture(texture2, texCoord - vec2(0.0, texelSize.y * i)).rgb * weights[i];
            }
        } else {
            result = vec3(texture(texture2, texCoord).rgb);
        }
        
        //mix with original texture
        vec3 pix1 = texture(texture1, mirroredCoord).rgb;
        vec3 pix2 = texture(texture1, vec2(mirroredCoord.x + texelSize.x, mirroredCoord.y)).rgb;
        vec3 pix3 = texture(texture1, vec2(mirroredCoord.x, mirroredCoord.y + texelSize.y)).rgb;
        vec3 pix4 = texture(texture1, vec2(mirroredCoord.x + texelSize.x, mirroredCoord.y + texelSize.y)).rgb;
        vec3 rawPixel = mapColor((pix1 + pix2 + pix3 + pix4) / 4);
        FragColor = vec4(rawPixel + result, 1.0);
    }
}
