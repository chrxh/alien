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
    // For debugging: just render a semi-transparent white quad to see if anything shows up
    // Later we'll sample from the texture atlas properly
    FragColor = vec4(1.0, 1.0, 1.0, 0.5);
    
    // Sample from the overlay texture
    // vec4 textColor = texture(overlayTexture, gQuadCoord);
    
    // // Only show if there's actually text content (alpha > 0)
    // if (textColor.a < 0.01) {
    //     discard;
    // }
    
    // // Output the text color
    // FragColor = textColor;
}
