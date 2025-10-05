#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    // Sample the input texture (output from metaballs effect)
    vec4 color = texture(inputTexture, texCoord);
    
    // Calculate brightness as thickness for subsurface scattering
    float brightness = dot(color.rgb, vec3(0.333, 0.333, 0.333));
    
    // Subsurface scattering simulation
    // Thicker areas (brighter) scatter less light, thinner areas scatter more
    vec2 texelSize = 1.0 / viewportSize;
    
    // Sample neighboring pixels for light scattering
    vec3 scattered = vec3(0.0);
    float totalWeight = 0.0;
    
    // Scatter radius based on inverse thickness (brighter = thicker = less scatter)
    // Scale with zoom: at high zoom (close up), reduce effect; at low zoom (far away), increase effect
    // Zoom typically ranges from ~0.5 (far) to ~50+ (close)
    float zoomScale = max(0.5, min(10.0, sqrt(zoom) / 5.0));
    float scatterRadius = mix(12.0, 1.0, brightness) * zoomScale;
    int radius = int(ceil(scatterRadius));
    
    int scanRadius = max(1, radius / 3);
    int scanDist = 3;
    if(radius < 3) {
        scanDist = 1;
    }
    // Subsurface scattering kernel - samples in a circular pattern
    for (int y = -scanRadius; y <= scanRadius; y++) {
        for (int x = -scanRadius; x <= scanRadius; x++) {
            float dist = length(vec2(x, y));
            if (dist <= scatterRadius) {
                vec2 offset = vec2(x, y) * texelSize * scanDist;
                vec4 sample = texture(inputTexture, texCoord + offset);
                
                // Weight based on distance and thickness
                // Thinner areas allow more light to penetrate from neighbors
                // Reduced brightness dampening for stronger effect
                float weight = exp(-dist / scatterRadius) * (1.0 - brightness * 0.3);
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
    // Much more scattering in thinner (darker) areas - increased from 0.5 to 0.85
    float scatterMix = (1.0 - brightness) * 0.85;
    vec3 finalColor = mix(color.rgb, scattered, scatterMix);
    
    // Add pronounced translucent glow effect
    // Simulate light passing through thin areas - increased from 0.3 to 0.7
    float glowIntensity = (1.0 - brightness) * 0.7;
    vec3 glowColor = finalColor * 1.5; // Much stronger brightness boost for glow
    finalColor = mix(finalColor, glowColor, glowIntensity);

    FragColor = vec4(scattered, 1.0f);//vec4(finalColor, 1.0f);
}
