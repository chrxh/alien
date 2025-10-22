#version 330 core
out vec4 FragColor;

in vec3 gColor;
in vec2 gQuadCoord;
in vec2 gWorldPos;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;
uniform sampler2D overlayTexture;

void main()
{
    // Sample from the overlay texture
    // For now, use a simple approach: cell type determines which part of texture to sample
    vec4 textColor = texture(overlayTexture, gQuadCoord);
    
    // Only show if there's actually text content (alpha > 0)
    if (textColor.a < 0.01) {
        discard;
    }
    
    // Output the text color
    FragColor = textColor;
}
