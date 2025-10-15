#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform vec2 viewportSize;
uniform float zoom;
uniform int mode;

void main()
{
    vec4 color1 = texture(inputTexture1, texCoord);
    vec4 color2 = texture(inputTexture2, texCoord);

    if (mode == 0) {
        float brightness = clamp(dot(color2.rgb, vec3(1.0)), 0.0, 1.0);
        vec3 finalColor = mix(color1.rgb * 1.0, color2.rgb, brightness);
        FragColor = vec4(finalColor, 1.0f);
        // vec3 mergedColor = clamp(color1.rgb * 0.7 + color2.rgb * 0.3, 0.0, 1.0);
        // FragColor = vec4(mergedColor, 1.0f);
    } else if (mode == 1) {
        // float brightness = clamp(dot(color2.rgb, vec3(1.0)), 0.0, 1.0);
        // vec3 finalColor = mix(color1.rgb * 0.7, color2.rgb, brightness);
        // FragColor = vec4(finalColor, 1.0f);
    } else {
        FragColor = vec4(0.0);
    }
}
