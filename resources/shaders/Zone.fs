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
    
    // Simple alpha based on distance for soft edges
    float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    
    // Reduce overall alpha to make zones semi-transparent
    alpha *= 0.3;
    
    FragColor = vec4(vColor, alpha);
}
