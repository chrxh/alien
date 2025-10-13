#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    vec2 texelSize = 1.0 / viewportSize;
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    float blurRadius = zoom / 24.0;
    
    // Dynamic blur based on radius
    int radius = max(1, int(ceil(blurRadius)));
    
    // Apply vertical Gaussian blur
    for (int y = -radius; y <= radius; y++) {
        float distance = float(y);
        // Gaussian weight calculation
        float weight = exp(-0.5 * (distance * distance) / (blurRadius * blurRadius));
        vec2 offset = vec2(0.0, distance * texelSize.y);
        result += texture(inputTexture1, texCoord + offset) * weight;
        totalWeight += weight;
    }
    
    // Normalize by total weight
    FragColor = result / totalWeight;
}
