#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int aShapeType;
layout (location = 3) in float aDimension1;
layout (location = 4) in float aDimension2;

out vec3 vColor;
out vec2 vWorldPos;
flat out int vShapeType;
out float vDimension1;
out float vDimension2;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;

void main()
{
    vColor = aColor;
    vWorldPos = aPos;
    vShapeType = aShapeType;
    vDimension1 = aDimension1;
    vDimension2 = aDimension2;
    
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);
    
    // Set point size based on shape dimensions and zoom
    // Use max dimension to ensure shape fits in point
    float maxDim = (aShapeType == 0) ? aDimension1 * 2.0 : max(aDimension1, aDimension2);
    gl_PointSize = maxDim * zoom + 4.0;  // Add 4 pixels for anti-aliasing
}
