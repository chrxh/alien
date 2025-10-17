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
    
    // Efficient blur using wider sampling (simulates downsampling effect)
    float blurRadius = zoom * strength;
    
    // Clamp radius to reasonable range for performance
    int radius = max(1, min(int(ceil(blurRadius * 0.5)), 12));
    
    vec4 result = vec4(0.0);
    float totalWeight = 0.0;
    
    // Apply vertical Gaussian blur with 2x step size (downsampling effect)
    for (int y = -radius; y <= radius; y++) {
        float distance = float(y);
        // Gaussian weight calculation
        float weight = exp(-0.5 * (distance * distance) / (blurRadius * blurRadius * 0.25));
        
        // Use 2x step to effectively sample at half resolution
        vec2 offset = vec2(0.0, distance * texelSize.y * 2.0);
        
        result += texture(inputTexture1, texCoord + offset) * weight;
        totalWeight += weight;
    }
    
    // Normalize by total weight
    FragColor = result / totalWeight;
}
