#version 330 core
out vec4 FragColor;

in vec3 vColor;
in vec2 vWorldPos;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;

void main()
{
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float dist = length(coord);
    
    // Discard pixels outside the circle
    if (dist > 0.5) {
        discard;
    }
    
    // Calculate the world position of this pixel
    // Point size is in screen pixels, we need to convert back to world space
    vec2 pixelOffset = (coord * 2.0) * (radius * 2.0) / zoom;
    vec2 pixelWorldPos = vWorldPos + pixelOffset;
    
    // Clip pixels outside world boundaries (pixel-wise clipping)
    if (pixelWorldPos.x < 0.0 || pixelWorldPos.x > worldSize.x ||
        pixelWorldPos.y < 0.0 || pixelWorldPos.y > worldSize.y) {
        discard;
    }
    
    // Simple alpha based on distance for soft edges
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    
    // Reduce overall alpha to make zones semi-transparent
    alpha *= 0.3;
    
    FragColor = vec4(vColor, alpha);
}
