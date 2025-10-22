#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int isActive;

out vec3 vertexColor;
out float vertexZPos;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
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
    
    // Triangles are rendered at the back (no bias added)
    
    gl_Position = vec4(ndc, normalizedZ - 1.0, 1.0);
    
    // Pass color and z-position to geometry shader
    vertexColor = aColor;
    vertexZPos = aPos.z;
}
