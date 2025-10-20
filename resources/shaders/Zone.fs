#version 330 core
out vec4 FragColor;

in vec3 vColor;
in vec2 vWorldPos;
flat in int vShapeType;
in float vDimension1;
in float vDimension2;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;

void main()
{
    // Calculate the world position of this pixel
    // Point size is in screen pixels, we need to convert back to world space
    vec2 coord = gl_PointCoord - vec2(0.5, 0.5);
    float maxDim = (vShapeType == 0) ? vDimension1 * 2.0 : max(vDimension1, vDimension2);
    vec2 pixelOffset = (coord * 2.0) * (maxDim * zoom + 4.0) / zoom;
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
        
        // Discard pixels outside the circle
        if (normalizedDist > vDimension1) {
            discard;
        }
        
        // Smooth edge with anti-aliasing
        float edgeStart = vDimension1 - 2.0 / zoom;
        float edgeEnd = vDimension1;
        alpha = 1.0 - smoothstep(edgeStart, edgeEnd, normalizedDist);
        
    } else {
        // Rectangular shape
        vec2 halfSize = vec2(vDimension1 * 0.5, vDimension2 * 0.5);
        vec2 absOffset = abs(pixelOffset);
        
        // Discard pixels outside the rectangle
        if (absOffset.x > halfSize.x || absOffset.y > halfSize.y) {
            discard;
        }
        
        // Smooth edges with anti-aliasing
        float edgeThickness = 2.0 / zoom;
        float distToEdgeX = halfSize.x - absOffset.x;
        float distToEdgeY = halfSize.y - absOffset.y;
        float distToEdge = min(distToEdgeX, distToEdgeY);
        
        alpha = smoothstep(0.0, edgeThickness, distToEdge);
    }
    
    // Reduce overall alpha to make zones semi-transparent
    alpha *= 0.3;
    
    FragColor = vec4(vColor, alpha);
}
