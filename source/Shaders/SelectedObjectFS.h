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
const float RAD_TO_DEG = 180.0 / PI;
const float AlphaForRestriction = 0.1;
const float AlphaForCircle = 0.3;

void main()
{
    // Calculate distance from center using quad coordinates
    float dist = length(gQuadCoord);
    
    if (gHasSignalRestriction == 0) {

        // Circle radius in normalized quad space (0.5 = edge)
        float outerRadius = 0.5;
        float middleRadius = 0.4;
        float innerRadius = middleRadius - 3.0 / zoom; // Thin circle (2 pixels)

        if (dist > outerRadius || dist < innerRadius) {
            discard;
        }

        FragColor = vec4(1.0, 1.0, 1.0, dist > middleRadius ? AlphaForRestriction : AlphaForCircle);
    }

    // If signal restriction is active, check if the current pixel is within the allowed angle range
    else if (gHasSignalRestriction == 1) {
    
        // Circle radius in normalized quad space (0.5 = edge)
        float outerRadius = 0.5;
        float middleRadius = 0.4;
        float innerRadius = middleRadius - 3.0 / zoom; // Thin circle (2 pixels)

        if (dist > outerRadius || dist < innerRadius) {
            discard;
        }

        // Calculate angle of current pixel relative to center
        // Note: gQuadCoord.x is horizontal (right = positive), gQuadCoord.y is vertical (down = positive)
        // We need to convert this to the same coordinate system used in the preview (0 degrees = up, clockwise)
        float pixelAngle = atan(gQuadCoord.x, -gQuadCoord.y) * RAD_TO_DEG;
        
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
        
        float alpha = AlphaForRestriction;
        if (dist <= middleRadius) {
            alpha = AlphaForCircle;
        } else if (!inRange) {
            discard;
        }

        FragColor = vec4(1.0, 1.0, 1.0, alpha);
    }
}
)";
}
