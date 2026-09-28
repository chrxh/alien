#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DetonationEventFS = R"(
#version 330 core

// Flash of a detonation, rendered into a separate buffer and later added to the scene. It lights up the scene with relief lighting,
// soft shadows and light shafts and vanishes within a few frames. Sizes are measured in detonator radii, times in seconds.

in vec2 localPos;
flat in vec2 centerPixelPos;
flat in float blastRadius;
flat in float age;

out vec4 FragColor;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float lifetime;

#define LIGHT_REACH 10.0
#define NUM_SHADOW_SAMPLES 12

const vec3 FlashColor = vec3(0.8, 0.9, 1.35);

float luminance(vec3 color)
{
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float occupancy(vec3 color)
{
    return smoothstep(0.02, 0.25, luminance(color));
}

void main()
{
    vec2 pos = localPos / blastRadius;
    float dist = length(pos);
    if (dist > LIGHT_REACH) {
        discard;
    }

    float flash = (age < 0.015 ? 1.0 : exp(-(age - 0.015) / 0.02)) * (1.0 - smoothstep(0.5 * lifetime, lifetime, age));
    float reachFade = 1.0 - smoothstep(0.5, 1.0, dist / LIGHT_REACH);
    vec3 light = FlashColor * flash * reachFade;

    vec2 texel = 1.0 / viewportSize;
    vec2 uv = gl_FragCoord.xy * texel;
    float pixelsPerRadius = blastRadius * zoom;
    vec3 scene = texture(inputTexture1, uv).rgb;

    float offset = 1.5;
    float heightLeft = luminance(texture(inputTexture1, uv - vec2(offset * texel.x, 0.0)).rgb);
    float heightRight = luminance(texture(inputTexture1, uv + vec2(offset * texel.x, 0.0)).rgb);
    float heightDown = luminance(texture(inputTexture1, uv - vec2(0.0, offset * texel.y)).rgb);
    float heightUp = luminance(texture(inputTexture1, uv + vec2(0.0, offset * texel.y)).rgb);
    vec2 gradient = vec2(heightRight - heightLeft, heightUp - heightDown) / (2.0 * offset) * zoom;
    vec3 normal = normalize(vec3(-gradient * 1.5, 1.0));
    vec3 toLight = normalize(vec3(-localPos, blastRadius * 0.7));
    float diffuse = max(dot(normal, toLight), 0.0);
    float specular = pow(max(dot(normal, normalize(toLight + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);

    // March towards the detonation and accumulate the bodies in between. Bodies next to the blast are torn apart and cast no shadow.
    vec2 toCenter = centerPixelPos - gl_FragCoord.xy;
    float distanceInPixels = length(toCenter);
    float startFraction = min(1.0, zoom / max(distanceInPixels, 1e-3));
    float stepLength = distanceInPixels * (1.0 - startFraction) / float(NUM_SHADOW_SAMPLES) / zoom;
    float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float opticalDepth = 0.0;
    for (int i = 0; i < NUM_SHADOW_SAMPLES; ++i) {
        float fraction = mix(startFraction, 1.0, (float(i) + jitter) / float(NUM_SHADOW_SAMPLES));
        float sampleDist = (1.0 - fraction) * distanceInPixels / pixelsPerRadius;
        float clearing = smoothstep(0.9, 1.7, sampleDist);
        opticalDepth += occupancy(texture(inputTexture1, (gl_FragCoord.xy + toCenter * fraction) * texel).rgb) * clearing;
    }
    float shadow = exp(-opticalDepth * stepLength * 0.9);

    vec3 core = light * (60.0 * exp(-dist * dist * 2.0) + 10.0 * exp(-dist * dist * 0.12));
    vec3 surfaceLight = light * 40.0 / (1.0 + dist * dist * 0.25) * shadow * (scene * (0.3 + 1.7 * diffuse) + occupancy(scene) * specular * 0.8);
    vec3 airLight = light * 6.0 / (1.0 + dist * dist * 0.25) * mix(1.0, shadow, 0.85);
    FragColor = vec4(core + surfaceLight + airLight, 0.0);
}
)";
}
