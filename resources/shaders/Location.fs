#version 330 core
out vec4 FragColor;

in vec3 gColor;
in vec2 gWorldPos;
flat in int gShapeType;
in float gDimension1;
in float gDimension2;
in float gFadeoutRadius;
in float gOpacity;
in vec2 gQuadCoord;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;

void main()
{
    // Calculate the world position of this pixel using quad coordinates
    // gQuadCoord ranges from -0.5 to 0.5
    float maxDim = (gShapeType == 0) ? (gDimension1 + gFadeoutRadius) * 2.0 : max(gDimension1 + gFadeoutRadius, gDimension2 + gFadeoutRadius);
    float padding = 4.0 / zoom;
    float halfSize = maxDim * 0.5 + padding;
    vec2 pixelOffset = gQuadCoord * 2.0 * halfSize;
    vec2 pixelWorldPos = gWorldPos + pixelOffset;
    
    // Clip pixels outside world boundaries (pixel-wise clipping)
    if (pixelWorldPos.x < 0.0 || pixelWorldPos.x > worldSize.x ||
        pixelWorldPos.y < 0.0 || pixelWorldPos.y > worldSize.y) {
        discard;
    }
    
    float alpha = 0.0;
    
    if (gShapeType == 0) {
        // Circular shape
        // Calculate distance from center using pixel offset in world space
        float distFromCenter = length(pixelOffset);
        
        // Discard pixels outside the circle + fadeout radius
        if (distFromCenter > gDimension1 + gFadeoutRadius) {
            discard;
        }
        
        // Calculate alpha based on distance
        if (distFromCenter <= gDimension1) {
            // Inside core radius - full opacity with anti-aliasing at edge
            float edgeStart = gDimension1 - 2.0 / zoom;
            float edgeEnd = gDimension1;
            alpha = 1.0;// - smoothstep(edgeStart, edgeEnd, distFromCenter);
        } else {
            // In fadeout zone - smooth transition from core to edge
            float fadeoutStart = gDimension1;
            float fadeoutEnd = gDimension1 + gFadeoutRadius;
            alpha = 1.0 - smoothstep(fadeoutStart, fadeoutEnd, distFromCenter);
        }
        
    } else {
        // Rectangular shape
        vec2 halfSize = vec2(gDimension1 * 0.5, gDimension2 * 0.5);
        vec2 absOffset = abs(pixelOffset);
        
        // Calculate distance to rectangle edge (positive = outside, negative = inside)
        vec2 distanceFromRect = vec2(
            max(0.0, absOffset.x - halfSize.x),
            max(0.0, absOffset.y - halfSize.y)
        );
        float distToEdge = length(distanceFromRect);
        
        // Discard pixels outside the rectangle + fadeout radius
        if (distToEdge > gFadeoutRadius) {
            discard;
        }
        
        // Calculate alpha based on distance to edge
        if (distToEdge > 0.0) {
            // Outside rectangle, in fadeout zone
            alpha = 1.0 - smoothstep(0.0, gFadeoutRadius, distToEdge);
        } else {
            // Inside rectangle - full opacity with anti-aliasing 
            alpha = 1.0f;
        }
    }
    
    // Apply layer opacity to the calculated alpha
    alpha *= gOpacity;
    
    FragColor = vec4(gColor, alpha);
}
