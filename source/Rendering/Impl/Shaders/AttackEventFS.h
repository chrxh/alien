#pragma once

#include <string_view>

namespace Shaders
{
    // A morsel is a torn off piece of tissue with a ragged outline, some grain and a faint halo
    std::string_view const AttackEventFS = R"(
#version 450
in vec3 morselLight;
in vec2 morselCoord;
flat in float morselOpacity;
flat in float morselSeed;
flat in float morselPixels;
out vec4 FragColor;

const float RaggedOutline = 0.4;
const float HaloGlow = 0.22;

float hash(vec2 position)
{
    vec3 mixed = fract(vec3(position.xyx) * 0.1031);
    mixed += dot(mixed, mixed.yzx + 33.33);
    return fract((mixed.x + mixed.y) * mixed.z);
}

float valueNoise(vec2 position)
{
    vec2 cell = floor(position);
    vec2 offset = fract(position);
    vec2 weight = offset * offset * (3.0 - 2.0 * offset);
    return mix(mix(hash(cell), hash(cell + vec2(1.0, 0.0)), weight.x), mix(hash(cell + vec2(0.0, 1.0)), hash(cell + vec2(1.0, 1.0)), weight.x), weight.y);
}

void main()
{
    // Coordinates in radii of the morsel
    float distance = length(morselCoord);
    vec2 direction = morselCoord / max(distance, 1e-5);
    float outline = 1.0 + RaggedOutline * (valueNoise(direction * 1.6 + morselSeed * 31.0) - 0.5);
    float pixel = 1.0 / max(morselPixels, 1e-3);
    float inside = clamp(0.5 - (distance - outline) / max(0.8 * pixel, 0.02), 0.0, 1.0);

    float grain = valueNoise(morselCoord * 2.2 + morselSeed * 19.0);
    float rimDistance = (distance - 0.86 * outline) / max(0.15, 0.6 * pixel);
    float rim = exp(-rimDistance * rimDistance) * smoothstep(1.5, 4.0, morselPixels);
    vec3 tissue = morselLight * (1.05 + 0.4 * grain) + mix(morselLight, vec3(dot(morselLight, vec3(0.333))), 0.4) * 0.35 * rim;

    // The halo fades out before the edge of the quad
    float quadEdge = 1.0 - smoothstep(1.3, 1.75, max(abs(morselCoord.x), abs(morselCoord.y)));
    float halo = exp(-max(distance - outline, 0.0) / max(0.5, 1.5 * pixel)) * (1.0 - inside) * HaloGlow * quadEdge;

    // The light is added to the target, the alpha channel holds the coverage
    FragColor = vec4((tissue * inside + morselLight * halo) * morselOpacity, inside * morselOpacity);
}
)";
}
