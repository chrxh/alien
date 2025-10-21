#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform vec2 viewportSize;
uniform float zoom;
uniform vec3 background;
uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform bool gridLines;
uniform float rectLowerRightX;

// Modulo function that handles negative numbers correctly
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
    vec3 finalColor;
    if (worldPos.x >= 0.0 && worldPos.x <= worldSize.x &&
        worldPos.y >= 0.0 && worldPos.y <= worldSize.y) {
        // Inside world boundaries - render background color
        finalColor = background;
    } else {
        // Outside world boundaries - render black
        finalColor = vec3(0.0, 0.0, 0.0);
    }
    
    // Add grid lines if enabled
    if (gridLines) {
        // Calculate grid parameters (matching CUDA implementation)
        float viewWidth = max(1.0, rectLowerRightX - rectUpperLeft.x);
        float pixelInWorldSize = viewWidth / worldSize.x;
        float gridDistance = pow(10.0, floor(log(viewWidth) / log(10.0))) / 10.0;
        float maxGridDistance = viewWidth / 10.0;
        float gridRemainder = (maxGridDistance - gridDistance) / maxGridDistance;
        
        // Coarse grid lines (larger spacing)
        {
            float distanceX = modulo(worldPos.x + gridDistance / 2.0, gridDistance) - gridDistance / 2.0;
            float distanceY = modulo(worldPos.y + gridDistance / 2.0, gridDistance) - gridDistance / 2.0;
            
            if (abs(distanceX) <= pixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceX) * zoom / 10.0) * gridRemainder * 0.7;
                finalColor += vec3(viewDistance);
            }
            if (abs(distanceY) <= pixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceY) * zoom / 10.0) * gridRemainder * 0.7;
                finalColor += vec3(viewDistance);
            }
        }
        
        // Fine grid lines (smaller spacing, 1/10th of coarse grid)
        {
            float distanceX = modulo(worldPos.x + gridDistance / 20.0, gridDistance / 10.0) - gridDistance / 20.0;
            float distanceY = modulo(worldPos.y + gridDistance / 20.0, gridDistance / 10.0) - gridDistance / 20.0;
            
            if (abs(distanceX) <= pixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceX) * zoom / 10.0) * (1.0 - gridRemainder) * 0.7;
                finalColor += vec3(viewDistance);
            }
            if (abs(distanceY) <= pixelInWorldSize * 8.0) {
                float viewDistance = max(0.0, 0.1 - abs(distanceY) * zoom / 10.0) * (1.0 - gridRemainder) * 0.7;
                finalColor += vec3(viewDistance);
            }
        }
    }
    
    FragColor = vec4(finalColor, 1.0);
}
