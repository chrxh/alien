#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D objectBackgroundTexture;
uniform sampler2D objectForegroundTexture;
uniform sampler2D screenBackgroundTexture;

void main()
{
    vec4 objectBackgroundColor = texture(objectBackgroundTexture, texCoord);
    vec4 objectForegroundColor = texture(objectForegroundTexture, texCoord);
    vec4 screenBackground = texture(screenBackgroundTexture, texCoord);
    
    // Merge the first two textures
    vec3 mergedColor = objectBackgroundColor.rgb * 0.5 + objectForegroundColor.rgb * 0.5;
    
    // Calculate brightness of merged result
    float brightness = clamp(dot(mergedColor, vec3(1.0)), -1.0, 1.0);
    
    // Use brightness as alpha to blend with screen background
    vec3 finalColor = mix(screenBackground.rgb, mergedColor, brightness);

    FragColor = vec4(finalColor, 1.0f);
}
