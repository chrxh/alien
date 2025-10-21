#version 330 core
out vec4 FragColor;

in vec2 texCoord;

uniform vec2 viewportSize;
uniform float zoom;
uniform vec3 background;
uniform vec2 worldSize;
uniform vec2 rectUpperLeft;

void main()
{
    // Convert texture coordinates to screen position (in pixels)
    vec2 screenPos = texCoord * viewportSize;
    
    // Convert screen position to world position
    vec2 relativePos = screenPos / zoom;
    vec2 worldPos = relativePos + rectUpperLeft;
    
    // Check if world position is within world boundaries
    if (worldPos.x >= 0.0 && worldPos.x <= worldSize.x &&
        worldPos.y >= 0.0 && worldPos.y <= worldSize.y) {
        // Inside world boundaries - render background color
        FragColor = vec4(background, 1.0);
    } else {
        // Outside world boundaries - render black
        FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
}
