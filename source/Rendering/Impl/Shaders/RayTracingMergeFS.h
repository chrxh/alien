#pragma once

#include <string_view>

namespace Shaders
{
    // Merges the ray tracing, its attack events and the rasterized objects, which replace the tissue when zoomed out. Like the tissue, the
    // rasterized objects absorb the background in high dynamic range and scatter the energy light. The absorption in the output of the ray
    // tracing is therefore decoded, combined with the one of the rasterized objects and encoded again, see getForeground in
    // Shaders/RayTracingTissueCS.h.
    std::string_view const RayTracingMergeFS = R"(
#version 450
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;
uniform sampler2D inputTexture4;
uniform vec2 viewportSize;
uniform float zoom;
uniform float objectGain;
uniform float tissueVisibility;
uniform float backgroundScale;

// Brightness of the rasterized objects above which they cover the background completely
const float CoverageBrightness = 0.3;

// The rasterized objects are whitened with their energy, while the tissue shows their saturated colors. The strongest component is kept.
const float WhiteRemoval = 0.6;

// The absorption of the tissue and how it scatters the energy light, see Shaders/RayTracingTissueCS.h
const float Saturation = 0.9;
const float Absorption = 2.2;
const float TissueEnergyLight = 0.8;

// The tone mapping of Shaders/RayTracingCommonCS.h
vec3 toneMapping(vec3 color)
{
    vec3 mapped = clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
    return pow(mapped, vec3(1.0 / 1.6));
}

vec3 inverseToneMapping(vec3 color)
{
    vec3 mapped = min(pow(clamp(color, 0.0, 1.0), vec3(1.6)), vec3(0.98));
    vec3 qa = mapped * 2.43 - 2.51;
    vec3 qb = mapped * 0.59 - 0.03;
    vec3 qc = mapped * 0.14;
    return (-qb - sqrt(max(qb * qb - 4.0 * qa * qc, vec3(0.0)))) / (2.0 * qa);
}

vec3 removeWhite(vec3 color)
{
    float maxComponent = max(color.r, max(color.g, color.b));
    float white = WhiteRemoval * min(color.r, min(color.g, color.b));
    return (color - white) * maxComponent / max(maxComponent - white, 1e-4);
}

// The background is added after the tone mapping, so an absorption is encoded such that adding the background yields the tone mapped
// absorbed background. The light and the background are given in the scale of the bloom.
vec3 encodeAbsorption(vec3 light, vec3 background)
{
    vec3 darkening = toneMapping(background) - toneMapping(max(background + light, vec3(0.0)));
    return mix(-inverseToneMapping(darkening), light, step(vec3(0.0), light));
}

vec3 decodeAbsorption(vec3 light, vec3 background)
{
    vec3 absorption = inverseToneMapping(toneMapping(background) - toneMapping(max(-light, vec3(0.0)))) - background;
    return mix(absorption, light, step(vec3(0.0), light));
}

void main()
{
    float bloomScale = 1.0 / backgroundScale;
    vec3 background = inverseToneMapping(texture(inputTexture4, texCoord).rgb);
    vec4 rayTracingOutput = texture(inputTexture1, texCoord);
    vec3 rayTracing = decodeAbsorption(rayTracingOutput.rgb * bloomScale, background);
    vec3 attackEvents = texture(inputTexture2, texCoord).rgb * objectGain * tissueVisibility;
    vec3 objects = texture(inputTexture3, texCoord).rgb * objectGain;

    float brightness = max(objects.r, max(objects.g, objects.b));
    vec3 hue = objects / max(brightness, 1e-4);
    vec3 color = mix(vec3(dot(hue, vec3(0.2126, 0.7152, 0.0722))), hue, Saturation);
    vec3 transmittance = exp(-Absorption * (vec3(1.0) - color * 0.85));
    float coverage = clamp(brightness / CoverageBrightness, 0.0, 1.0);

    // The ray tracing passes on the strength of the energy light in its alpha channel
    vec3 energyLight = coverage * color * rayTracingOutput.a * TissueEnergyLight;
    vec3 objectLight = (removeWhite(objects) + energyLight) * bloomScale - background * coverage * (vec3(1.0) - transmittance);

    vec3 light = rayTracing + objectLight * (1.0 - tissueVisibility);
    FragColor = vec4(encodeAbsorption(light, background) * backgroundScale + attackEvents, 1.0);
}
)";
}
