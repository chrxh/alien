#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture;
uniform vec2 viewportSize;
uniform float blurRadius;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    
    // Dynamic blur based on radius
    int radius = int(ceil(blurRadius));
    
    // Apply horizontal Gaussian blur
    for (int x = -radius; x <= radius; x++) {
        float distance = float(x);
        // Gaussian weight calculation
        float weight = exp(-0.5 * (distance * distance) / (blurRadius * blurRadius));
        vec2 offset = vec2(distance * texelSize.x, 0.0);
        result += texture(inputTexture, texCoord + offset) * weight;
        totalWeight += weight;
    }
    
    // Normalize by total weight
    FragColor = result / totalWeight;
}
