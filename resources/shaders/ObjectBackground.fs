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
    } else if (zoom > 1.0) {
        float startValue = zoom / 4.0 * 0.5;
        float endValue = 0.5 * zoom * zoom / 4.0;
        alpha = 1.0 - smoothstep(min(startValue, endValue), max(startValue, endValue), dist);
    }
    else {
        if (dist > 0.3) {
            discard;
        }
        alpha = zoom * zoom * 0.2;
    }
    FragColor = vec4(vColor, alpha);
}
