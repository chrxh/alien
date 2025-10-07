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
    
    // Interpret brightness as height map
    float height = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Calculate normal from height map using Sobel operator
    vec2 texelSize = 1.0 / viewportSize;
    
    // Sample neighboring heights
    float h_left = dot(texture(inputTexture, texCoord + vec2(-texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float h_right = dot(texture(inputTexture, texCoord + vec2(texelSize.x, 0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float h_top = dot(texture(inputTexture, texCoord + vec2(0.0, -texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    float h_bottom = dot(texture(inputTexture, texCoord + vec2(0.0, texelSize.y)).rgb, vec3(0.299, 0.587, 0.114));
    
    // Calculate gradients (heightmap derivatives)
    float dx = (h_right - h_left) * 0.5;
    float dy = (h_bottom - h_top) * 0.5;
    
    // Compensate for zoom level - at high zoom, multiply by zoom to maintain bump illusion
    // At low zoom, the effect is already visible, so we use a logarithmic scale
    float zoomCompensation = mix(1.0, zoom, clamp(zoom / 10.0, 0.0, 1.0));
    
    // Strength factor to amplify the normal calculation
    float normalStrength = 5.0 * zoomCompensation;
    dx *= normalStrength;
    dy *= normalStrength;
    
    // Construct normal vector
    vec3 normal = normalize(vec3(-dx, -dy, 1.0));
    
    // Fixed light direction (from top-left, slightly forward)
    vec3 lightDir = normalize(vec3(0.5, 0.5, 1.0));
    
    // Calculate diffuse lighting (Lambertian)
    float diffuse = max(dot(normal, lightDir), 0.0);
    
    // Add ambient light to prevent completely dark areas
    float ambient = 0.3;
    float lighting = ambient + diffuse * 0.7;
    
    // Apply lighting to the color
    vec3 litColor = color.rgb * lighting;
    
    // Add specular highlight for more pronounced effect
    vec3 viewDir = vec3(0.0, 0.0, 1.0); // Camera looking straight down
    vec3 halfDir = normalize(lightDir + viewDir);
    float specular = pow(max(dot(normal, halfDir), 0.0), 32.0);
    
    // Add specular highlight
    vec3 specularColor = vec3(1.0) * specular * 0.5;
    litColor += specularColor;
    
    // Enhance contrast based on height
    // Higher areas get more light, lower areas get less
    float heightBoost = height * 0.3;
    litColor += litColor * heightBoost;
    
    FragColor = vec4(litColor, 1.0);
}
