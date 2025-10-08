#version 330 core
out vec4 FragColor;

in vec3 vColor;

uniform bool smoothCircles;

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
    
    // Apply smooth anti-aliasing at edges
    //float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    //float alpha = cos(dist / 180.0 * 3.14159 * 90.0);
    if (smoothCircles) {
        float angle = atan(coord.y, coord.x) * 180.0 / PI;
        angle += 45.0;
        if (angle > 180.0) {
            angle -= 360.0;
        }
        float factor = min(1.0, 40.0 / (abs(angle) + 1.0));
        //float alpha = (1.0 - smoothstep(0.3, 0.5, dist)) * factor;
        float alpha = (1.0 - smoothstep(0.3, 0.5, dist)) * smoothstep(0.0, 0.5, dist) * factor;
        FragColor = vec4(vColor, alpha);
    } else {
        float alpha = cos(dist / 180.0 * 3.14159 * 90.0);
        FragColor = vec4(vColor, alpha);
    }
}
