#pragma once

#include <string_view>

namespace Shaders
{
    // Shows the fluid particles as gas, a participating medium that absorbs the background, scatters the light of a lamp at the upper left
    // and of glowing cells and glows weakly by itself. Glowing particles appear as sparks inside. The lamp light reaches each point through
    // the gas towards the lamp, which gives the clouds bright edges on one side and shaded cores. Turbulence adds detail that evolves with
    // the simulation time and stays still while the simulation pauses.
    std::string_view const RayTracingGasShadingCS = R"(
const vec2 GasLampDirection = vec2(-0.7071, -0.7071);
const float GasExtinction = 2.5;
const float GasLampIntensity = 2.0;
const float GasAmbientLight = 0.2;
const float GasEmission = 0.5;
const float GasSparkIntensity = 4.0;
const float GasBioluminescence = 0.6;
const float GasShadowDistance = 5.0;
const int GasShadowSteps = 8;
const float GasShadowStrength = 0.8;
const float GasGlowIntensity = 1.2;
const int GasGlowTaps = 16;

const float GasDetailFrequency = 0.2;
const float GasDetailContrast = 1.0;
const float GasDetailWarp = 2.5;
const int GasDetailOctaves = 5;
const float GasDetailLacunarity = 2.03;

// Octaves with wavelengths below a few pixels would flicker, they fade to the mean value instead
const float GasDetailMinWavelengthInPixels = 3.0;

float getGasFbm(vec2 position, float numOctaves)
{
    float result = 0.0;
    float amplitude = 0.5;
    float amplitudeSum = 0.0;
    for (int i = 0; i < GasDetailOctaves; ++i) {
        float visibility = clamp(numOctaves - float(i), 0.0, 1.0);
        result += amplitude * mix(0.5, valueNoise(position), visibility);
        amplitudeSum += amplitude;
        position = position * GasDetailLacunarity + vec2(17.1, 9.3);
        amplitude *= 0.5;
    }
    return result / amplitudeSum;
}

float getGasTurbulence(vec2 position)
{
    float coarsestWavelengthInPixels = 1.0 / (GasDetailFrequency * getPixelSize());
    float numOctaves = log2(coarsestWavelengthInPixels / GasDetailMinWavelengthInPixels) / log2(GasDetailLacunarity) + 1.0;
    numOctaves = clamp(numOctaves, 0.0, float(GasDetailOctaves));

    vec2 scaledPosition = position * GasDetailFrequency;
    float phase = frame.gasGrid.w;
    vec2 warp = vec2(getGasFbm(scaledPosition + vec2(phase, 0.0), numOctaves), getGasFbm(scaledPosition + vec2(5.2, 1.3 - phase), numOctaves));
    return getGasFbm(scaledPosition + GasDetailWarp * (warp - 0.5) + vec2(0.0, 0.7 * phase), numOctaves);
}

// Light scattered once more spreads around the gas as a soft glow. The taps lie on a spiral and sample the already smooth density, the
// sharp sparks would appear repeatedly.
vec3 getGasGlow(vec2 position)
{
    float radius = frame.gasSpread.y;
    vec3 sum = vec3(0.0);
    float weightSum = 0.0;
    for (int i = 0; i < GasGlowTaps; ++i) {
        float fraction = (float(i) + 0.5) / float(GasGlowTaps);
        float angle = float(i) * 2.39996;
        float weight = exp(-2.0 * fraction);
        sum += weight * sampleGas(position + radius * sqrt(fraction) * vec2(cos(angle), sin(angle))).rgb;
        weightSum += weight;
    }
    return sum / weightSum;
}

vec3 shadeGas(vec2 position, vec3 background, vec3 scatteredLight)
{
    vec3 glow = GasGlowIntensity * getGasGlow(position) + GasSparkIntensity * sampleSparks(position);
    vec4 gas = sampleGas(position);
    if (gas.a < 1e-3) {
        return background + glow;
    }
    vec3 color = gas.rgb / gas.a;
    float turbulence = max(1.0 + GasDetailContrast * (2.0 * getGasTurbulence(position) - 1.0), 0.0);
    float opticalDepth = GasExtinction * gas.a * turbulence;
    float transmittance = exp(-opticalDepth);

    // Emitted light partly gets absorbed by the gas itself, the fraction that escapes decreases in dense gas
    float escapingFraction = (1.0 - transmittance) / max(opticalDepth, 1e-4);
    vec3 emission = GasEmission * gas.rgb * turbulence * escapingFraction;

    float shadowDensity = 0.0;
    float stepLength = GasShadowDistance / float(GasShadowSteps);
    for (int i = 1; i <= GasShadowSteps; ++i) {
        shadowDensity += sampleGas(position + GasLampDirection * (stepLength * float(i))).a;
    }
    float lampTransmittance = exp(-GasExtinction * GasShadowStrength * shadowDensity * stepLength);

    vec3 light = vec3(GasLampIntensity * lampTransmittance + GasAmbientLight) + GasBioluminescence * scatteredLight;
    return background * transmittance + color * light * (1.0 - transmittance) + emission + glow;
}
)";
}
