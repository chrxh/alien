#version 330 core
out vec4 FragColor;

in vec3 vColor;
in vec2 vWorldPos;
flat in int vShapeType;
in float vDimension1;
in float vDimension2;
in float vFadeoutRadius;
in float vOpacity;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;

void main()
{
    // Calculate the world position of this pixel
    // Point size is in screen pixels, we need to convert back to world space
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float maxDim = (vShapeType == 0) ? (vDimension1 + vFadeoutRadius) * 2.0 : max(vDimension1 + vFadeoutRadius, vDimension2 + vFadeoutRadius);
    vec2 pixelOffset = (coord * 1.0) * (maxDim * zoom + 4.0) / zoom;
    vec2 pixelWorldPos = vWorldPos + pixelOffset;
    
    // Clip pixels outside world boundaries (pixel-wise clipping)
    if (pixelWorldPos.x < 0.0 || pixelWorldPos.x > worldSize.x ||
        pixelWorldPos.y < 0.0 || pixelWorldPos.y > worldSize.y) {
        discard;
    }
    
    float alpha = 0.0;
    
    if (vShapeType == 0) {
        // Circular shape
        float dist = length(coord);
        float normalizedDist = dist * (maxDim * zoom + 4.0) / zoom;
        
        // Discard pixels outside the circle + fadeout radius
        if (normalizedDist > vDimension1 + vFadeoutRadius) {
            discard;
        }
        
        // Calculate alpha based on distance
        if (normalizedDist <= vDimension1) {
            // Inside core radius - full opacity with anti-aliasing at edge
            float edgeStart = vDimension1 - 2.0 / zoom;
            float edgeEnd = vDimension1;
            alpha = 1.0 - smoothstep(edgeStart, edgeEnd, normalizedDist);
        } else {
            // In fadeout zone - smooth transition from core to edge
            float fadeoutStart = vDimension1;
            float fadeoutEnd = vDimension1 + vFadeoutRadius;
            alpha = 1.0 - smoothstep(fadeoutStart, fadeoutEnd, normalizedDist);
        }
        
    } else {
        // Rectangular shape
        vec2 halfSize = vec2(vDimension1 * 0.5, vDimension2 * 0.5);
        vec2 absOffset = abs(pixelOffset);
        
        // Calculate distance to rectangle edge (positive = outside, negative = inside)
        vec2 distanceFromRect = vec2(
            max(0.0, absOffset.x - halfSize.x),
            max(0.0, absOffset.y - halfSize.y)
        );
        float distToEdge = length(distanceFromRect);
        
        // Discard pixels outside the rectangle + fadeout radius
        if (distToEdge > vFadeoutRadius) {
            discard;
        }
        
        // Calculate alpha based on distance to edge
        if (distToEdge > 0.0) {
            // Outside rectangle, in fadeout zone
            alpha = 1.0 - smoothstep(0.0, vFadeoutRadius, distToEdge);
        } else {
            // Inside rectangle - full opacity with anti-aliasing at edge
            float edgeThickness = 2.0 / zoom;
            float distToEdgeX = halfSize.x - absOffset.x;
            float distToEdgeY = halfSize.y - absOffset.y;
            float distToEdgeInside = min(distToEdgeX, distToEdgeY);
            alpha = smoothstep(0.0, edgeThickness, distToEdgeInside);
        }
    }
    
    // Apply layer opacity to the calculated alpha
    alpha *= vOpacity;
    
    FragColor = vec4(vColor, alpha);
}
