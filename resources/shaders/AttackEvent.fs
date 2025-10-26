#version 330 core
in vec3 fragColor;
in vec2 lineCoord;
out vec4 FragColor;

uniform float zoom;

void main()
{
    // Create dashed pattern
    // Scale the dash pattern based on zoom level
    float dashSize = 0.05 * zoom;
    float gapSize = 0.05 * zoom;
    float patternLength = dashSize + gapSize;
    
    // Calculate position along the line
    float linePos = lineCoord.x;
    float pattern = mod(linePos, patternLength);
    
    // Discard fragments in the gap
    if (pattern > dashSize) {
        discard;
    }
    
    // Use the color from geometry shader
    FragColor = vec4(fragColor, 1.0);
}
