#version 330 core
out vec4 FragColor;

in vec3 vColor;
in float vRadius;

void main()
{
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float dist = length(coord);
    
    // Discard pixels outside the circle
    if (dist > 0.5) {
        discard;
    }
    
    // Apply smooth anti-aliasing at edges
    float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
    
    FragColor = vec4(vColor, alpha);
}
