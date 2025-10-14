#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aZPos;

out vec3 fragColor;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform vec2 viewportSize;

#define PI 3.1415926538

void main()
{
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);
    
    // Apply simple lighting based on z-position (simulating depth-based lighting)
    // This provides a consistent lighting effect without needing a geometry shader
    // Use a simple ambient + diffuse model where higher z = more light
    float lightIntensity = 0.8 + 0.2 * (aZPos / 10.0);
    lightIntensity = clamp(lightIntensity, 0.8, 1.0);
    
    // Apply lighting to vertex color
    fragColor = aColor * lightIntensity;
}
