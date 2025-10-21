#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform vec2 viewportSize;
uniform float zoom;
uniform vec3 background;
uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform int gridLines;

// GLSL modulo function that matches CUDA Math::modulo behavior
float modulo(float a, float b) {
    return a - b * floor(a / b);
}

void main()
{
    // Convert texture coordinates to screen position (in pixels)
    vec2 screenPos = texCoord * viewportSize;
    
    // Convert screen position to world position
    vec2 relativePos = screenPos / zoom;
    vec2 worldPos = relativePos + rectUpperLeft;
    
    // Check if world position is within world boundaries
    vec3 color;
    if (worldPos.x >= 0.0 && worldPos.x <= worldSize.x &&
        worldPos.y >= 0.0 && worldPos.y <= worldSize.y) {
        // Inside world boundaries - render background color
        color = background;
    } else {
        // Outside world boundaries - render black
        color = vec3(0.0, 0.0, 0.0);
    }
    
    // Add grid lines if enabled
    if (gridLines != 0 && worldPos.x >= 0.0 && worldPos.x <= worldSize.x &&
        worldPos.y >= 0.0 && worldPos.y <= worldSize.y) {
        
        // Calculate grid parameters based on view width (matches CUDA logic)
        float viewWidth = max(1.0, worldSize.x);  // Simplified - could be rectLowerRight.x - rectUpperLeft.x
        float pixelInWorldSize = viewWidth / worldSize.x;
        float gridDistance = pow(10.0, floor(log(viewWidth) / log(10.0))) / 10.0;
        float maxGridDistance = viewWidth / 10.0;
        float gridRemainder = (maxGridDistance - gridDistance) / maxGridDistance;
        
        // First grid set (coarse grid)
        float distanceX1 = modulo(worldPos.x + gridDistance / 2.0, gridDistance) - gridDistance / 2.0;
        float distanceY1 = modulo(worldPos.y + gridDistance / 2.0, gridDistance) - gridDistance / 2.0;
        
        if (abs(distanceX1) <= pixelInWorldSize * 8.0) {
            float viewDistance = max(0.0, 0.1 - abs(distanceX1) * zoom / 10.0) * gridRemainder * 0.7;
            color += vec3(viewDistance);
        }
        if (abs(distanceY1) <= pixelInWorldSize * 8.0) {
            float viewDistance = max(0.0, 0.1 - abs(distanceY1) * zoom / 10.0) * gridRemainder * 0.7;
            color += vec3(viewDistance);
        }
        
        // Second grid set (fine grid)
        float distanceX2 = modulo(worldPos.x + gridDistance / 20.0, gridDistance / 10.0) - gridDistance / 20.0;
        float distanceY2 = modulo(worldPos.y + gridDistance / 20.0, gridDistance / 10.0) - gridDistance / 20.0;
        
        if (abs(distanceX2) <= pixelInWorldSize * 8.0) {
            float viewDistance = max(0.0, 0.1 - abs(distanceX2) * zoom / 10.0) * (1.0 - gridRemainder) * 0.7;
            color += vec3(viewDistance);
        }
        if (abs(distanceY2) <= pixelInWorldSize * 8.0) {
            float viewDistance = max(0.0, 0.1 - abs(distanceY2) * zoom / 10.0) * (1.0 - gridRemainder) * 0.7;
            color += vec3(viewDistance);
        }
    }
    
    FragColor = vec4(color, 1.0);
}
