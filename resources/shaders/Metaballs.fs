#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;

void main()
{
    // Sample the input texture
    vec4 color = texture(inputTexture1, texCoord);
    
    // For metaballs effect, use the luminance/brightness as the density
    // This works even when alpha is 1 by treating bright colors as "dense" areas
    float luminance = color.r + color.g + color.b;//dot(color.rgb, vec3(0.299, 0.587, 0.114));
    
    // Also incorporate alpha for smoother edges from anti-aliasing
    float density = luminance * color.a;
    
    // Apply metaballs threshold with smoothstep for smooth blob-like edges
    float threshold = 0.3;
    float smoothRange = 0.35;
    float metaball = smoothstep(threshold - smoothRange, threshold + smoothRange, density);
    
    // Output with metaball effect - modulate color intensity
    vec3 outputColor = color.rgb * metaball;
    
    FragColor = clamp(vec4(outputColor, 1.0f), 0.0, 1.0);
}
