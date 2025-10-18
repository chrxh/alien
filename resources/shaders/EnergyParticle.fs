#version 330 core
out vec4 FragColor;

in vec3 vColor;
uniform float zoom;

void main()
{
    // Calculate distance from center of point
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float dist = length(coord);
    
    // Discard pixels outside the circle
    if (dist > 0.5) {
        discard;
    }
    
    // Simple falloff for alpha based on distance from center
    float alpha;
    if (zoom > 4.0) {
        alpha = 1.0 - smoothstep(0.0, 0.5, dist);
    } else if (zoom > 1.0) {
        float startValue = zoom / 4.0 * 0.5;
        float endValue = 0.5 * zoom * zoom / 4.0;
        alpha = 1.0 - smoothstep(min(startValue, endValue), max(startValue, endValue), dist);
    } else {
        if (dist > 0.3) {
            discard;
        }
        alpha = zoom * zoom * 0.3;
    }
    
    FragColor = vec4(vColor, alpha);
}
