#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedObjectFS = R"(
#version 450
out vec4 FragColor;

in vec2 gWorldPos;
in vec2 gPixelOffset;
flat in float gCircleRadius;
flat in float gSmoothSpread;
flat in float gDetail;

uniform float zoom;
uniform vec2 worldSize;
uniform float radius;
uniform float renderScale;

const float AlphaForFadeout = 0.1;
const float AlphaForCircle = 0.3;
const float OuterRadius = 0.5;
const float MiddleRadius = 0.4;
const float FullAlphaZoom = 8.0;
const float PI = 3.1415926538;

float getOpticalDepth(float alpha)
{
    return -log(1.0 - alpha);
}

float getInnerRadius(float screenZoom)
{
    return MiddleRadius - 3.0 / screenZoom;  // Thin circle (2 pixels)
}

float getCircleDepth(float dist, float screenZoom)
{
    if (dist > OuterRadius || dist < getInnerRadius(screenZoom)) {
        return 0.0;
    }
    float zoomFadeout = min(1.0, screenZoom / FullAlphaZoom);
    return getOpticalDepth((dist > MiddleRadius ? AlphaForFadeout : AlphaForCircle) * zoomFadeout);
}

// The circles alias into moire patterns at low zoom, and their minimum size brightens dense selections when zooming out.
// Below the detail zoom each object therefore spreads the optical depth of its circle at full alpha over a Gaussian.
// The depths of overlapping objects add up, so the color of a selection stays the same at all low zoom levels.
float getSmoothDepth(float pixelDist)
{
    float innerRadius = max(0.0, getInnerRadius(FullAlphaZoom));
    float circleRadiusWithoutMinimum = gCircleRadius * zoom / radius;
    float depthIntegral = 4.0 * PI * circleRadiusWithoutMinimum * circleRadiusWithoutMinimum
        * ((OuterRadius * OuterRadius - MiddleRadius * MiddleRadius) * getOpticalDepth(AlphaForFadeout)
           + (MiddleRadius * MiddleRadius - innerRadius * innerRadius) * getOpticalDepth(AlphaForCircle));
    float spread = gSmoothSpread;
    float gaussianMass = 1.0 - exp(-4.5);
    return depthIntegral * exp(-pixelDist * pixelDist / (2.0 * spread * spread)) / (2.0 * PI * spread * spread * gaussianMass);
}

void main()
{
    float pixelDist = length(gPixelOffset);
    float circleDepth = gDetail > 0.0 ? getCircleDepth(pixelDist / (2.0 * gCircleRadius), zoom / renderScale) : 0.0;
    float smoothDepth = gDetail < 1.0 && pixelDist <= 3.0 * gSmoothSpread ? getSmoothDepth(pixelDist) : 0.0;
    float depth = mix(smoothDepth, circleDepth, gDetail);
    if (depth <= 0.0) {
        discard;
    }
    FragColor = vec4(1.0, 1.0, 1.0, 1.0 - exp(-depth));
}
)";
}
