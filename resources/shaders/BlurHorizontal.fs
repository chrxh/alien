#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float strength;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    
    // Use fixed kernel size for better performance
    // Precomputed Gaussian weights for 13-tap kernel (sigma ~= 4.0)
    const float weights[7] = float[7](
        0.19638, 0.29675, 0.09450, 0.01038, 0.00038, 0.00000, 0.00000
    );
    
    vec4 result = texture(inputTexture1, texCoord) * weights[0];
    
    // Scale offset by strength and zoom for adaptive blur
    float offset = strength * max(1.0, zoom * 0.1);
    
    // Apply horizontal blur with fixed 13-tap kernel
    for (int i = 1; i < 7; i++) {
        float off = float(i) * texelSize.x * offset;
        result += texture(inputTexture1, texCoord + vec2(off, 0.0)) * weights[i];
        result += texture(inputTexture1, texCoord - vec2(off, 0.0)) * weights[i];
    }
    
    FragColor = result;
}
