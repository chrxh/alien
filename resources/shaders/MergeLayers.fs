#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;
uniform vec2 viewportSize;
uniform float zoom;
uniform int numTextures;

void main()
{
    vec4 color1 = texture(inputTexture1, texCoord);
    vec4 color2 = texture(inputTexture2, texCoord);
    vec4 color3 = texture(inputTexture3, texCoord);

    float brightness2 = clamp(dot(color2.rgb, vec3(2.0)), 0.0, 1.0);
    vec3 finalColor = mix(color1.rgb, color2.rgb, brightness2);

    if (numTextures >= 3) {
        float brightness3 = clamp(dot(color3.rgb, vec3(2.0)), 0.0, 1.0);
        finalColor = mix(finalColor.rgb, color3.rgb, brightness3);
    }

    FragColor = vec4(finalColor, 1.0f);
}
