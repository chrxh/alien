#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int state;

out vec3 vColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;

void main()
{
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    
    // Normalize z to depth range [0, 1] - smaller z values should be closer (rendered in front)
    // Z values range from approximately 0 to 1000 (creature ID % 1000 + random offset)
    float normalizedZ = aPos.z / 1000.0;
    normalizedZ = clamp(normalizedZ, 0.0, 1.0);
    
    // Cells are rendered in front of lines (apply negative bias to bring forward)
    normalizedZ -= 0.0002;
    normalizedZ = clamp(normalizedZ, 0.0, 1.0);
    
    gl_Position = vec4(ndc, normalizedZ, 1.0);

    if (state == 1 && zoom > 6.0f) {
        vColor = mix(aColor, vec3(1.0), 0.1);
        gl_PointSize = radius * 0.3;
    } else if (state == 2 && zoom > 6.0f) {
        vColor = mix(aColor, vec3(1.0), 0.2);
        gl_PointSize = radius * 0.4;
    } else {
        vColor = aColor;
        gl_PointSize = radius * 0.2;
    }
}
