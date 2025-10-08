#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture;
uniform sampler2D objectTextureSmall;

void main()
{
    // Sample the input texture (output from Fresnel effect)
    vec4 color = texture(inputTexture, texCoord);
    
    // Sample the original small object texture
    vec4 originalColor = texture(objectTextureSmall, texCoord);
    
    // Additively blend with original small object texture
    vec3 finalColor = color.rgb * 0.5 + originalColor.rgb * 0.5;

    FragColor = vec4(finalColor, 1.0f);
}
