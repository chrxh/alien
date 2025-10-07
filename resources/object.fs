#version 330 core
out vec4 FragColor;

in vec3 vColor;

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
    float alpha = cos(dist / 180.0 * 3.14159 * 90.0);
//    float alpha = (1.0 - smoothstep(0.3, 0.5, dist)) * smoothstep(0.1, 0.3, dist);
    
    FragColor = vec4(vColor, alpha);
}
