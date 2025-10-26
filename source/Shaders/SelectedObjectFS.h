#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedObjectFS = R"(
#version 330 core
out vec4 FragColor;

in vec2 gWorldPos;
in vec2 gQuadCoord;
flat in int gHasSignalRestriction;
flat in float gStartAngle;
flat in float gEndAngle;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;

const float PI = 3.14159265359;
const float DEG_TO_RAD = PI / 180.0;

void main()
{
    // Calculate distance from center using quad coordinates
    float dist = length(gQuadCoord);
    
    // Circle radius in normalized quad space (0.5 = edge)
    float outerRadius = 0.5;
    float innerRadius = outerRadius - 3.0 / zoom; // Thin circle (2 pixels)
    
    // Discard pixels outside the outer radius
    if (dist > outerRadius) {
        discard;
    }
    
    // Discard pixels inside the inner radius (creating a hollow circle)
    if (dist < innerRadius) {
        discard;
    }
    
    // Anti-aliasing for smooth edges
    float outerEdge = smoothstep(outerRadius - 1.5 / zoom, outerRadius, dist);
    float innerEdge = smoothstep(innerRadius, innerRadius + 1.5 / zoom, dist);
    float alpha = (1.0 - outerEdge) * innerEdge;
    
    // If signal restriction is active, check if the current pixel is within the allowed angle range
    if (gHasSignalRestriction == 1) {
        // Calculate angle of current pixel relative to center
        // Note: gQuadCoord.x is horizontal (right = positive), gQuadCoord.y is vertical (down = positive)
        // We need to convert this to the same coordinate system used in the preview (0 degrees = up, clockwise)
        float pixelAngle = atan(gQuadCoord.x, -gQuadCoord.y) / DEG_TO_RAD;
        
        // Normalize to 0-360 range
        if (pixelAngle < 0.0) {
            pixelAngle += 360.0;
        }
        
        // Check if pixel is within the signal restriction angle range
        bool inRange;
        if (gStartAngle <= gEndAngle) {
            // Normal case: range doesn't wrap around
            inRange = (pixelAngle >= gStartAngle && pixelAngle <= gEndAngle);
        } else {
            // Wrapping case: range wraps around 0/360
            inRange = (pixelAngle >= gStartAngle || pixelAngle <= gEndAngle);
        }
        
        if (!inRange) {
            discard;
        }
    }
    
    // White circle
    FragColor = vec4(1.0, 1.0, 1.0, alpha);
}
)";
}
