#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const OrganicSurfaceFS = R"(
#version 330 core

// Reshapes the object layer into rounded, translucent bodies. Every shading term is neutral on flat
// interior areas, so only the boundaries are formed. Applied techniques:
//
//   Domain warping (fBm value noise)   irregular contour
//   Coverage averaging                 hides the triangulation
//   Morphological rounding             surface tension of the silhouette
//   Thickness based saturation         volume
//   Screen space normals               relief from the gradient of the density field
//   Half Lambert diffuse               soft shading
//   Fresnel rim                        translucent membrane
//   Subsurface scattering              light leaving thin regions
//   Curvature ambient occlusion        crevices between adjoining bodies
//   Blinn-Phong specular               wet sheen
//   fBm mottling                       granular interior

out vec4 FragColor;

in vec2 texCoord;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform vec2 rectUpperLeft;
uniform float zoom;

uniform float effectStrength;
uniform float warpStrength;
uniform float warpFrequency;
uniform float smoothingStrength;
uniform float depthStrength;
uniform float roundingStrength;
uniform float reliefStrength;
uniform float shadingStrength;
uniform float membraneStrength;
uniform float scatterStrength;
uniform float specularStrength;
uniform float cavityStrength;
uniform float grainStrength;
uniform float grainFrequency;

#define PI 3.1415926538
#define NUM_DIRECTIONS 12

// Noise coordinates are wrapped by this value to keep them small enough for float precision
#define NOISE_PERIOD 256.0

float hash(vec2 pos)
{
    vec3 mixed = fract(vec3(pos.xyx) * 0.1031);
    mixed += dot(mixed, mixed.yzx + 33.33);
    return fract((mixed.x + mixed.y) * mixed.z);
}

float valueNoise(vec2 pos)
{
    vec2 cell = floor(pos);
    vec2 offset = fract(pos);
    vec2 weight = offset * offset * (3.0 - 2.0 * offset);
    float lowerLeft = hash(cell);
    float lowerRight = hash(cell + vec2(1.0, 0.0));
    float upperLeft = hash(cell + vec2(0.0, 1.0));
    float upperRight = hash(cell + vec2(1.0, 1.0));
    return mix(mix(lowerLeft, lowerRight, weight.x), mix(upperLeft, upperRight, weight.x), weight.y);
}

float fbm(vec2 pos, int octaves)
{
    float result = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < octaves; ++i) {
        result += amplitude * valueNoise(pos);
        pos *= 2.0;
        amplitude *= 0.5;
    }
    return result / (1.0 - pow(0.5, float(octaves)));
}

float density(vec3 color)
{
    return smoothstep(0.02, 0.25, color.r + color.g + color.b);
}

void main()
{
    vec3 originalColor = texture(inputTexture1, texCoord).rgb;
    if (effectStrength <= 0.0) {
        FragColor = vec4(originalColor, 1.0);
        return;
    }

    vec2 texelSize = 1.0 / viewportSize;

    // The texture coordinate runs upwards while rectUpperLeft is the top world corner, hence the flip
    vec2 worldPos = vec2(texCoord.x, 1.0 - texCoord.y) * viewportSize / zoom + rectUpperLeft;
    float cellRadiusInPixels = max(zoom * 0.5, 1.0);

    vec2 samplePos = texCoord;
    if (warpStrength > 0.0) {
        float warpFrequencyInWorld = min(warpFrequency, zoom / 16.0);
        vec2 warpPos = mod(worldPos * warpFrequencyInWorld, NOISE_PERIOD);
        vec2 warp = vec2(fbm(warpPos, 2), fbm(warpPos + vec2(5.2, 1.3), 2)) - 0.5;
        samplePos += warp * warpStrength * cellRadiusInPixels * texelSize * effectStrength;
    }

    vec3 centerColor = texture(inputTexture1, samplePos).rgb;
    float centerDensity = density(centerColor);

    float innerRadius = clamp(cellRadiusInPixels * 0.35, 1.0, 40.0);
    float outerRadius = clamp(cellRadiusInPixels * 0.9, 2.0, 80.0);
    vec2 gradient = vec2(0.0);
    float innerDensity = 0.0;
    float outerDensity = 0.0;
    vec3 innerColor = vec3(0.0);
    vec3 outerColor = vec3(0.0);
    for (int i = 0; i < NUM_DIRECTIONS; ++i) {
        float angle = (float(i) + 0.5) * 2.0 * PI / float(NUM_DIRECTIONS);
        vec2 direction = vec2(cos(angle), sin(angle));

        vec3 innerSample = texture(inputTexture1, samplePos + direction * innerRadius * texelSize).rgb;
        vec3 outerSample = texture(inputTexture1, samplePos + direction * outerRadius * texelSize).rgb;
        float innerSampleDensity = density(innerSample);

        gradient += direction * innerSampleDensity;
        innerDensity += innerSampleDensity;
        outerDensity += density(outerSample);
        innerColor += innerSample;
        outerColor += outerSample;
    }
    gradient /= float(NUM_DIRECTIONS);
    innerDensity /= float(NUM_DIRECTIONS);
    outerDensity /= float(NUM_DIRECTIONS);
    innerColor /= float(NUM_DIRECTIONS);
    outerColor /= float(NUM_DIRECTIONS);

    vec3 blendedColor = (centerColor + innerColor + outerColor) / 3.0;
    vec3 result = mix(centerColor, blendedColor, smoothingStrength * innerDensity);

    // Only large bodies are rounded, small objects would be eroded away
    float coverage = 0.2 * centerDensity + 0.45 * innerDensity + 0.35 * outerDensity;
    float shape = smoothstep(0.3, 0.7, coverage);
    float largeBody = smoothstep(0.35, 0.7, outerDensity);
    float rounding = roundingStrength * largeBody;
    result = mix(result, innerColor, rounding * max(shape - centerDensity, 0.0));
    result *= mix(1.0, shape, rounding);

    float thickness = smoothstep(0.15, 0.85, innerDensity);
    float luminance = dot(result, vec3(0.299, 0.587, 0.114));
    vec3 deepColor = mix(vec3(luminance), result, 1.35) * 0.8;
    result = mix(result, deepColor, depthStrength * thickness);

    vec3 normal = normalize(vec3(-gradient * reliefStrength, 1.0));
    vec3 lightDirection = normalize(vec3(-0.55, 0.55, 0.45));
    vec3 halfVector = normalize(lightDirection + vec3(0.0, 0.0, 1.0));

    float diffuse = 0.5 + 0.5 * dot(normal, lightDirection);
    float flatDiffuse = 0.5 + 0.5 * lightDirection.z;
    result *= mix(1.0, diffuse / flatDiffuse, shadingStrength);

    float edge = centerDensity * (1.0 - thickness);
    float rim = pow(1.0 - normal.z, 1.5) * edge;
    result += rim * membraneStrength * centerColor;

    result += outerColor * scatterStrength * edge;

    float cavity = max(outerDensity - innerDensity, 0.0);
    result *= 1.0 - cavityStrength * cavity * centerDensity;

    float slope = smoothstep(0.05, 0.35, length(gradient));
    float specular = pow(max(dot(normal, halfVector), 0.0), 12.0);
    result += specular * specularStrength * slope * centerDensity;

    if (grainStrength > 0.0) {
        float grainFrequencyInWorld = min(grainFrequency, zoom / 4.0);
        float grain = fbm(mod(worldPos * grainFrequencyInWorld, NOISE_PERIOD), 2) - 0.5;
        result *= 1.0 + grain * grainStrength * thickness;
    }

    FragColor = vec4(mix(originalColor, max(result, vec3(0.0)), effectStrength), 1.0);
}
)";
}
