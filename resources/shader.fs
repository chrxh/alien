#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform sampler2D texture3;
uniform int phase;
uniform bool glowEffect;
uniform bool motionEffect;
uniform float motionBlurFactor;
uniform float brightness;
uniform float contrast;

// Rendered pixels per screen pixel. Greater than 1 while a picture is rendered with a higher resolution than the
// view. The glow radius is defined in texels, so it has to be stretched by this factor to keep its visual size.
uniform float renderScale = 1.0;

uniform float weights[14] = float[](0.15, 0.15, 0.09, 0.06, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02);

vec3 mapColor(vec3 color)
{
    return ((sqrt(color * 256.0 ) - 0.7) * contrast + 0.5) * brightness;
}

void main()
{
    vec2 texelSize = 1.0 / textureSize(texture1, 0);
    vec2 mirroredCoord = vec2(texCoord.x, 1.0 - texCoord.y);

    // Every weight covers `renderScale` texels instead of one. Averaging that many sub samples per weight keeps the
    // kernel dense, a single tap per weight would skip texels and make the glow band at higher resolutions.
    int subSamples = max(1, int(renderScale + 0.5));

    //horizontal blur
    if (phase == 0) {
        vec3 result;
        if (glowEffect) {
            result = vec3(texture(texture1, mirroredCoord).rgb * weights[0]);
            for (int i = 1; i < 14; ++i) {
                vec3 sum = vec3(0.0);
                for (int s = 0; s < subSamples; ++s) {
                    float offset = texelSize.x * (float(i) * renderScale - float(s));
                    sum += texture(texture1, mirroredCoord + vec2(offset, 0.0)).rgb;
                    sum += texture(texture1, mirroredCoord - vec2(offset, 0.0)).rgb;
                }
                result += sum / float(subSamples) * weights[i];
            }
        } else {
            result = vec3(texture(texture1, mirroredCoord).rgb);
        }
        result = mapColor(result);

        FragColor = vec4(result, 1.0);
    }

    //vertical blur
    if(phase == 1) {
        vec3 result;
        if (glowEffect) {
            result =
                vec3(texture(texture2, texCoord).rgb * weights[0]);
            for (int i = 1; i < 14; ++i) {
                vec3 sum = vec3(0.0);
                for (int s = 0; s < subSamples; ++s) {
                    float offset = texelSize.y * (float(i) * renderScale - float(s));
                    sum += texture(texture2, texCoord + vec2(0.0, offset)).rgb;
                    sum += texture(texture2, texCoord - vec2(0.0, offset)).rgb;
                }
                result += sum / float(subSamples) * weights[i];
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
        result = result + rawPixel;

        if (motionEffect) {
            result = result * motionBlurFactor + texture(texture3, texCoord).rgb * (1 - motionBlurFactor);
        }
        FragColor = vec4(result, 1.0);
    }

    //draw
    if (phase == 2) {
        FragColor = texture(texture1, texCoord);
    }
}
