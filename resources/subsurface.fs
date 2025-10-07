#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture;
uniform sampler2D objectTextureSmall;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    // Sample the input texture (output from metaballs effect)
    vec4 color = texture(inputTexture, texCoord);
    
    // Sample the original small object texture
    vec4 originalColor = texture(objectTextureSmall, texCoord);
    
    // Interpret brightness as height for bump mapping
    float height = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Calculate surface normals from height gradients using Sobel operator
    vec2 texelSize = 1.0 / viewportSize;
    
    // Sample neighboring heights
    float hL = dot(texture(inputTexture, texCoord + vec2(-texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float hR = dot(texture(inputTexture, texCoord + vec2(texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float hT = dot(texture(inputTexture, texCoord + vec2(0.0, -texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float hB = dot(texture(inputTexture, texCoord + vec2(0.0, texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    
    // Compute gradients with zoom compensation
    float zoomCompensation = mix(1.0, zoom, clamp(zoom / 10.0, 0.0, 1.0));
    float normalStrength = 8.0 * zoomCompensation;
    vec3 normal;
    normal.x = (hL - hR) * normalStrength;
    normal.y = (hT - hB) * normalStrength;
    normal.z = 1.0;
    normal = normalize(normal);
    
    // Fixed light direction from top-left
    vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
    
    // Zoom-dependent effect strength
    float effectStrength = 1.0 + zoom * 0.5;
    
    // Wrap-around diffuse lighting for subsurface scattering simulation
    // Light wraps around thin areas, simulating light penetration
    float wrapAmount = 0.5; // How much light wraps around
    float NdotL = dot(normal, lightDir);
    float wrapDiffuse = max(0.0, (NdotL + wrapAmount) / (1.0 + wrapAmount));
    
    // Thinner areas get more wrap-around effect (more translucent)
    float thinness = 1.0 - height;
    float wrapFactor = mix(0.0, 1.0, thinness);
    float standardDiffuse = max(0.0, NdotL);
    float diffuse = mix(standardDiffuse, wrapDiffuse, wrapFactor * 0.7);
    
    // Zoom-dependent diffuse strength
    diffuse = diffuse * 0.85 * effectStrength;
    
    // Zoom-dependent ambient light (less ambient at high zoom for better contrast)
    float ambient = mix(0.3, 0.15, clamp(zoom / 20.0, 0.0, 1.0));
    
    // Soft specular highlights using base color (organic materials)
    vec3 viewDir = vec3(0.0, 0.0, 1.0);
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfDir), 0.0), 8.0); // Low shininess for soft highlights
    float specularStrength = mix(0.15, 0.25, clamp(zoom / 20.0, 0.0, 1.0));
    vec3 specular = color.rgb * spec * specularStrength * effectStrength;
    
    // Translucency effect - thin areas glow slightly
    float translucency = thinness * 0.3;
    vec3 translucentGlow = color.rgb * translucency;
    
    // Combine lighting components
    vec3 litColor = color.rgb * (ambient + diffuse) + specular + translucentGlow;
    
    // Height-based brightness boost (zoom-dependent)
    float heightBoost = height * mix(0.3, 0.4, clamp(zoom / 20.0, 0.0, 1.0)) * effectStrength;
    litColor += color.rgb * heightBoost;
    
    // Additively blend with original small object texture
    vec3 finalColor = litColor + originalColor.rgb;
    
    FragColor = vec4(finalColor, 1.0);
}
