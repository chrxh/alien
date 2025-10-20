#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in int aShapeType;
layout (location = 3) in float aDimension1;
layout (location = 4) in float aDimension2;
layout (location = 5) in float aFadeoutRadius;
layout (location = 6) in float aOpacity;

out vec3 vColor;
out vec2 vWorldPos;
flat out int vShapeType;
out float vDimension1;
out float vDimension2;
out float vFadeoutRadius;
out float vOpacity;

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
    vFadeoutRadius = aFadeoutRadius;
    vOpacity = aOpacity;
    
    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);
    
    // Set point size based on shape dimensions, fadeout radius and zoom
    // Use max dimension + fadeout to ensure entire zone (including fadeout) fits in point
    float maxDim = (aShapeType == 0) ? (aDimension1 + aFadeoutRadius) * 2.0 : max(aDimension1 + aFadeoutRadius, aDimension2 + aFadeoutRadius);
    gl_PointSize = maxDim * zoom + 4.0;  // Add 4 pixels for anti-aliasing
}
