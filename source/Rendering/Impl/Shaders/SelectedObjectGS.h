#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const SelectedObjectGS = R"(
#version 450
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in vec2 vWorldPos[];

out vec2 gWorldPos;
out vec2 gPixelOffset;
flat out float gCircleRadius;
flat out float gSmoothSpread;
flat out float gDetail;

uniform vec2 viewportSize;
uniform float zoom;
uniform float radius;
uniform float renderScale;

const float MinDetailZoom = 4.0;
const float MaxDetailZoom = 8.0;
const float SmoothSpreadWorld = 0.55;
const float MinSmoothSpreadPixels = 0.6;

// Outputs are undefined after EmitVertex, so each vertex sets all of them
void emitVertex(vec4 center, vec2 pixelOffset, float circleRadius, float smoothSpread, float detail)
{
    gl_Position = vec4(center.xy + pixelOffset / viewportSize * vec2(2.0, -2.0), center.z, 1.0);
    gPixelOffset = pixelOffset;
    gWorldPos = vWorldPos[0];
    gCircleRadius = circleRadius;
    gSmoothSpread = smoothSpread;
    gDetail = detail;
    EmitVertex();
}

void main()
{
    // Thin circle at high zoom, smooth highlight below
    float circleRadius = radius * 0.35;
    float smoothSpread = max(SmoothSpreadWorld * zoom, MinSmoothSpreadPixels * renderScale);
    float detail = smoothstep(MinDetailZoom, MaxDetailZoom, zoom / renderScale);
    float halfSize = max(detail > 0.0 ? circleRadius : 0.0, detail < 1.0 ? 3.0 * smoothSpread : 0.0);

    vec4 center = gl_in[0].gl_Position;
    emitVertex(center, vec2(-halfSize, halfSize), circleRadius, smoothSpread, detail);
    emitVertex(center, vec2(halfSize, halfSize), circleRadius, smoothSpread, detail);
    emitVertex(center, vec2(-halfSize, -halfSize), circleRadius, smoothSpread, detail);
    emitVertex(center, vec2(halfSize, -halfSize), circleRadius, smoothSpread, detail);
    EndPrimitive();
}
)";
}
