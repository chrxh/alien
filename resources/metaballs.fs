#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture;
uniform vec2 viewportSize;

void main()
{
    // Sample the input texture
    vec4 color = texture(inputTexture, texCoord);
    
    // Metaballs effect: threshold the alpha channel to create blob-like appearance
    float threshold = 0.5;
    float alphaValue = color.a;
    
    // Apply threshold with smoothstep for smooth edges
    float metaball = smoothstep(threshold - 0.1, threshold + 0.1, alphaValue);
    
    // Output with metaball effect applied
    FragColor = vec4(color.rgb, metaball);
}
