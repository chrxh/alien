#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture;
uniform vec2 viewportSize;

void main()
{
    // Sample the input texture (output from metaballs effect)
    vec4 color = texture(inputTexture, texCoord);
    
    // Calculate brightness as thickness for subsurface scattering
    float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Subsurface scattering simulation
    // Thicker areas (brighter) scatter less light, thinner areas scatter more
    vec2 texelSize = 1.0 / viewportSize;
    
    // Sample neighboring pixels for light scattering
    vec3 scattered = vec3(0.0);
    float totalWeight = 0.0;
    
    // Scatter radius based on inverse thickness (brighter = thicker = less scatter)
    float scatterRadius = mix(3.0, 0.5, brightness);
    int radius = int(ceil(scatterRadius));
    
    // Subsurface scattering kernel - samples in a circular pattern
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float dist = length(vec2(x, y));
            if (dist <= scatterRadius) {
                vec2 offset = vec2(x, y) * texelSize;
                vec4 sample = texture(inputTexture, texCoord + offset);
                
                // Weight based on distance and thickness
                // Thinner areas allow more light to penetrate from neighbors
                float weight = exp(-dist / scatterRadius) * (1.0 - brightness * 0.7);
                scattered += sample.rgb * weight;
                totalWeight += weight;
            }
        }
    }
    
    // Normalize scattered light
    if (totalWeight > 0.0) {
        scattered /= totalWeight;
    }
    
    // Blend original color with scattered light
    // More scattering in thinner (darker) areas
    float scatterMix = (1.0 - brightness) * 0.5;
    vec3 finalColor = mix(color.rgb, scattered, scatterMix);
    
    // Add subtle translucent glow effect
    // Simulate light passing through thin areas
    float glowIntensity = (1.0 - brightness) * 0.3;
    vec3 glowColor = finalColor * 1.5; // Boosted brightness for glow
    finalColor = mix(finalColor, glowColor, glowIntensity);
    
    FragColor = vec4(finalColor, 1.0);
}
