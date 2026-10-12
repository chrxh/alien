#pragma once

#include <string_view>

namespace Shaders
{
    // Shows the fluids as a glowing nebula. Each fluid contributes as much light as its soft disk in the rasterized view and the clouds take
    // the outline of the rasterized fluids, so that brightness and shapes match the rasterized view at every zoom level. Thin gas glows in
    // saturated colors, dense gas towards warm white, and the edges of the clouds glow brighter. The gas also scatters the colored light of
    // the cells around it, see Shaders/RayTracingGasIrradianceCS.h.
    std::string_view const RayTracingGasShadingCS = R"(
// Light of a fluid in the rasterized view: a soft disk of radius 7.5 with a peak of 0.15
const float RasterDiskIntegral = 7.95;
const float RasterCloudFactor = 0.8;
const float RasterThresholdLow = 0.05;
const float RasterThresholdHigh = 0.55;

const float NebulaSaturation = 1.8;
const float NebulaMaxLuminanceGain = 1.3;
const vec3 NebulaCoreColor = vec3(1.0, 0.88, 0.75);
const float NebulaCoreWeight = 0.75;
const float NebulaCoreStart = 2.0;
const float NebulaCoreEnd = 8.0;
const float NebulaEdgeGlow = 1.5;

const float GasScattering = 0.6;
const float CellLightTint = 0.5;
const float CellLightLimit = 0.3;

float getThickness(vec2 position)
{
    vec4 clouds = sampleGasLayerSmooth(position, GasLayer_Clouds);
    return RasterDiskIntegral * max(clouds.r + clouds.g + clouds.b, 0.0);
}

// Gradient of the logarithmic thickness, which is high at the edges of the clouds
vec2 getThicknessGradient(vec2 position)
{
    float offset = max(frame.gasGrid.z, 0.5);
    float right = log(1.0 + getThickness(position + vec2(offset, 0.0)));
    float left = log(1.0 + getThickness(position - vec2(offset, 0.0)));
    float lower = log(1.0 + getThickness(position + vec2(0.0, offset)));
    float upper = log(1.0 + getThickness(position - vec2(0.0, offset)));
    return vec2(right - left, lower - upper) / (2.0 * offset);
}

vec3 shadeGas(vec2 position, vec3 background)
{
    vec4 clouds = sampleGasLayerSmooth(position, GasLayer_Clouds);
    float density = max(clouds.a, 0.0);
    if (density < 1e-4) {
        return background;
    }

    // The soft disks of the fluids add up and a threshold cuts the clouds as in the rasterized view, which dims the fluids when zoomed out
    // very far
    vec3 rasterClouds = RasterDiskIntegral * max(clouds.rgb, vec3(0.0));
    float thickness = rasterClouds.r + rasterClouds.g + rasterClouds.b;
    float coverage = smoothstep(RasterThresholdLow, RasterThresholdHigh, thickness);
    float zoomCorrection = min(1.0, 8.0 * frame.view.w);
    float brightness = RasterCloudFactor * maxComponent(rasterClouds) * coverage * zoomCorrection;

    // The saturated colors keep the luminance of the original colors up to a limit, so that the nebula is about as bright as the rasterized
    // fluids without overexposing deep blue
    vec3 color = clouds.rgb / density;
    vec3 hue = color / max(maxComponent(color), 1e-4);
    vec3 saturatedHue = pow(hue, vec3(NebulaSaturation));
    saturatedHue *= min(luminance(hue) / max(luminance(saturatedHue), 1e-4), NebulaMaxLuminanceGain);
    float core = NebulaCoreWeight * smoothstep(NebulaCoreStart, NebulaCoreEnd, thickness);
    vec3 nebulaColor = mix(saturatedHue, NebulaCoreColor, core);
    float edge = smoothstep(0.1, 0.5, length(getThicknessGradient(position))) * coverage;
    vec3 nebula = brightness * nebulaColor * (1.0 + NebulaEdgeGlow * edge);

    // The light of many cells, for example along chains of cells, would outshine the gas. It is limited softly, so that weak light stays
    // unchanged. The simulation parameter fluid illumination scales the result.
    float scattering = 1.0 - exp(-GasScattering * density);
    vec3 cellLight = sampleGasIrradiance(position) * mix(vec3(1.0), hue, CellLightTint) * scattering;
    float lightLevel = luminance(cellLight) / CellLightLimit;
    cellLight /= sqrt(1.0 + lightLevel * lightLevel);
    return background + nebula + frame.gasLighting.z * cellLight;
}
)";
}
