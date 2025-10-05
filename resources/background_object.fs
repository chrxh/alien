#version 330 core
out vec4 FragColor;

in vec3 vColor;
uniform float zoom;

#define PI 3.1415926538

void main()
{
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float dist = length(coord);
    
    // Discard pixels outside the circle
     if (dist > 0.5) {
         discard;
     }
    
    // Background objects - simple cosine falloff for alpha
    float alpha;
    if (zoom > 4.0) {
        alpha = cos(dist / 180.0 * 3.14159 * 90.0);
    } else {
        alpha = 1.0 - smoothstep(zoom / 4.0 * 0.35, 0.35 * min(1.0, zoom), dist);
    }
    FragColor = vec4(vColor, alpha);
}
