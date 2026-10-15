#pragma once

#include <string_view>

namespace Shaders
{
    // Merges the ray tracing, its attack events and the rasterized objects, which replace the tissue when zoomed out. Like the tissue, the
    // rasterized objects absorb the background in high dynamic range and scatter the light of the radiation sources. The absorption in the
    // output of the ray tracing is therefore decoded, combined with the one of the rasterized objects and encoded again, see getForeground
    // in Shaders/RayTracingTissueCS.h.
    std::string_view const RayTracingMergeFS = R"(
#version 450
out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform sampler2D inputTexture2;
uniform sampler2D inputTexture3;
uniform sampler2D inputTexture4;
uniform float objectGain;
uniform float tissueVisibility;
uniform float backgroundScale;

// Brightness of the rasterized objects above which they cover the background completely
const float CoverageBrightness = 0.3;

// The rasterized objects shine like the tissue, so that the colors do not change when zooming. Brighter objects would also shift the hues
// in the tone mapping.
const float ObjectBrightness = 0.88;

// The absorption of the tissue, see Shaders/RayTracingTissueCS.h
const float Saturation = 0.9;
const float Absorption = 2.2;

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

    // The ray tracing passes on the strength of the light of the radiation sources that the tissue scatters in its alpha channel
    vec3 sourceLight = coverage * color * rayTracingOutput.a;
    vec3 objectLight = (objects * ObjectBrightness + sourceLight) * bloomScale - background * coverage * (vec3(1.0) - transmittance);

    vec3 light = rayTracing + objectLight * (1.0 - tissueVisibility);
    FragColor = vec4(encodeAbsorption(light, background) * backgroundScale + attackEvents, 1.0);
}
)";
}
