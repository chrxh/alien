#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D backgroundObjectTexture;
uniform sampler2D foregroundObjectTexture;

void main()
{
    vec4 backgroundColor = texture(backgroundObjectTexture, texCoord);
    vec4 foregroundColor = texture(foregroundObjectTexture, texCoord);
    
    // Additively blend with original small object texture
    vec3 finalColor = backgroundColor.rgb * 0.5 + foregroundColor.rgb * 0.5;

    FragColor = vec4(finalColor, 1.0f);
}
