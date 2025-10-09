#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    vec4 color1 = texture(inputTexture1, texCoord);
    vec4 color2 = texture(inputTexture2, texCoord);
    //vec4 screenBackground = texture(screenBackgroundTexture, texCoord);
    
    // Merge the first two textures
    vec3 mergedColor = color1.rgb * 0.5 + color2.rgb * 0.5;
    
    FragColor = vec4(mergedColor, 1.0f);
    // Calculate brightness of merged result
    //float brightness = clamp(dot(mergedColor, vec3(1.0)), -1.0, 1.0);
    
    // Use brightness as alpha to blend with screen background
    //vec3 finalColor = mix(screenBackground.rgb, mergedColor, brightness);

    //FragColor = vec4(finalColor, 1.0f);
}
