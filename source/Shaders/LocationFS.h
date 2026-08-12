#pragma once

#include <string_view>

#include <EngineInterface/PerlinNoiseSource.h>

namespace Shaders
{
    std::string_view const LocationFS = R"(
#version 330 core
out vec4 FragColor;

in vec3 gColor;
in vec2 gWorldPos;
flat in int gShapeType;
in float gDimension1;
in float gDimension2;
in float gFadeoutRadius;
in float gOpacity;
flat in int gFieldType;
in float gFieldParam1;
in float gFieldParam2;
in vec2 gQuadCoord;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;
uniform bool borderlessRendering;

const int ForceField_None = 0;
const int ForceField_Radial = 1;
const int ForceField_Central = 2;
const int ForceField_Linear = 3;
const int ForceField_PerlinNoise = 4;

const float DegToRad = 0.0174532925;
const float CentralForceFieldOffset = 50.0;
const float MaxDarkening = 0.4;
)" PERLIN_NOISE_GLSL R"(

// Returns the height of the Perlin noise height map in [0, 1]
float perlinNoiseHeight(vec2 relPos)
{
    vec2 scaledPos = relPos / max(gFieldParam1, 0.1);
    int timeIndex = int(floor(gFieldParam2));
    float timeFraction = gFieldParam2 - float(timeIndex);
    float height0 = perlinNoise(scaledPos.x, scaledPos.y, perlinHash(timeIndex));
    float height1 = perlinNoise(scaledPos.x, scaledPos.y, perlinHash(timeIndex + 1));
    return mix(height0, height1, timeFraction) + 0.5;
}

// Returns the height map of the force field in [0, 1], normalized over the extent of the layer
float fieldHeight(vec2 relPos, float outerRadius)
{
    if (gFieldType == ForceField_Radial) {
        float height = sqrt(length(relPos) / outerRadius);
        return gFieldParam1 > 0.0 ? height : 1.0 - height;
    }
    if (gFieldType == ForceField_Central) {
        float distSquared = dot(relPos, relPos);
        float outerDistSquared = outerRadius * outerRadius;
        return log(1.0 + distSquared / CentralForceFieldOffset) / log(1.0 + outerDistSquared / CentralForceFieldOffset);
    }
    if (gFieldType == ForceField_Linear) {
        // The height map is the stream function of the field, hence its contour lines follow the flow
        float angle = gFieldParam1 * DegToRad;
        vec2 gradientDirection = vec2(-cos(angle), -sin(angle));
        return 0.5 + dot(relPos, gradientDirection) / (2.0 * outerRadius);
    }
    return perlinNoiseHeight(relPos);
}

void main()
{
    // Calculate the world position of this pixel using quad coordinates
    // gQuadCoord ranges from -0.5 to 0.5
    float maxDim = (gShapeType == 0) ? (gDimension1 + gFadeoutRadius) * 2.0 : max(gDimension1 + gFadeoutRadius * 2, gDimension2 + gFadeoutRadius * 2);
    // vec2 maxDim;
    // if (gShapeType == 0) {
    //     maxDim = vec2((gDimension1 + gFadeoutRadius) * 2.0, (gDimension1 + gFadeoutRadius) * 2.0);
    // } else {
    //     maxDim = vec2(gDimension1 + gFadeoutRadius * 2.0, gDimension2 + gFadeoutRadius * 2.0);
    // }
    float padding = 4.0 / zoom;
    float halfSize = maxDim * 0.5 + padding;
    vec2 pixelOffset = gQuadCoord * 2.0 * halfSize;
    vec2 pixelWorldPos = gWorldPos + pixelOffset;
    
    // Clip pixels outside world boundaries (pixel-wise clipping)
    if (!borderlessRendering) {
        if (pixelWorldPos.x < 0.0 || pixelWorldPos.x > worldSize.x ||
            pixelWorldPos.y < 0.0 || pixelWorldPos.y > worldSize.y) {
            discard;
        }
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

        // Discard when closer to other wrapped edge
        if (abs(pixelOffset.x) - halfSize.x > (worldSize.x - gDimension1) / 2) {
            discard;
        }
        if (abs(pixelOffset.y) - halfSize.y > (worldSize.y - gDimension2) / 2) {
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

    // Darken the color where the height map of the force field is high
    vec3 color = gColor;
    if (gFieldType != ForceField_None) {
        float outerRadius = (gShapeType == 0) ? gDimension1 + gFadeoutRadius : length(vec2(gDimension1, gDimension2)) * 0.5 + gFadeoutRadius;
        color *= 1.0 - MaxDarkening * clamp(fieldHeight(pixelOffset, max(outerRadius, 1.0)), 0.0, 1.0);
    }

    FragColor = vec4(color, alpha);
}
)";
}
