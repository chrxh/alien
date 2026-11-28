#pragma once

#include <string_view>

namespace Shaders
{
    // An attacked object loses morsels in its color, which drift along a slight curve into the attacker and shrink while it swallows them.
    // They follow one another in a stream that depends on the time step alone, so a paused simulation keeps its picture. Morsels smaller
    // than a pixel fade out instead of glittering.
    std::string_view const AttackEventGS = R"(
#version 450
layout (lines) in;
layout (triangle_strip, max_vertices = 12) out;

in vec3 vertexColor[];
in float vertexIntensity[];
in float vertexSeed[];
out vec3 morselLight;
out vec2 morselCoord;
flat out float morselOpacity;
flat out float morselSeed;
flat out float morselPixels;

uniform vec2 viewportSize;
uniform float zoom;
uniform float timestep;

const int MorselsPerAttack = 3;
const float MorselTravelTimesteps = 64.0;

// Sizes in world units
const float MorselStartRadius = 0.095;
const float MorselEndRadius = 0.045;
const float MorselDeparture = 0.2;

// The quad of a morsel leaves room for its halo, its size is given in radii
const float MorselQuadSize = 1.8;

// The morsels glow like the tissue of the prey
const float MorselGlow = 1.0;

vec2 transform(vec2 v)
{
    v = v / viewportSize * 2.0 - 1.0;
    return vec2(v.x, -v.y);
}

float hash(vec2 position)
{
    vec3 mixed = fract(vec3(position.xyx) * 0.1031);
    mixed += dot(mixed, mixed.yzx + 33.33);
    return fract((mixed.x + mixed.y) * mixed.z);
}

// Hue and brightness as the ray traced tissue derives them from the color of an object
vec3 getLight(vec3 color)
{
    float colorBrightness = max(color.r, max(color.g, color.b));
    vec3 hue = colorBrightness > 0.02 ? clamp(color / colorBrightness, 0.0, 1.0) : vec3(0.5);
    float brightness = min(colorBrightness, 1.0) + 0.5 * log(max(colorBrightness, 1.0));
    vec3 saturatedHue = mix(vec3(dot(hue, vec3(0.2126, 0.7152, 0.0722))), hue, 0.9);
    return saturatedHue * brightness * MorselGlow;
}

void emitMorsel(vec2 center, float radius, float opacity, vec3 light, float seed)
{
    float extent = radius * MorselQuadSize;
    vec2 corners[4] = vec2[4](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    for (int i = 0; i < 4; ++i) {
        gl_Position = vec4(transform(center + corners[i] * extent), 0.0, 1.0);
        morselLight = light;
        morselCoord = corners[i] * MorselQuadSize;
        morselOpacity = opacity;
        morselSeed = seed;
        morselPixels = radius;
        EmitVertex();
    }
    EndPrimitive();
}

void main()
{
    vec2 attacker = gl_in[0].gl_Position.xy;
    vec2 victim = gl_in[1].gl_Position.xy;
    float intensity = vertexIntensity[1];
    float seed = vertexSeed[1];
    vec2 axis = attacker - victim;
    float distance = length(axis);
    if (intensity <= 0.0 || distance < 1e-3) {
        return;
    }
    vec2 direction = axis / distance;
    vec2 normal = vec2(-direction.y, direction.x);
    vec2 start = victim + direction * min(MorselDeparture * zoom, 0.5 * distance);
    vec3 light = getLight(vertexColor[1]);

    for (int k = 0; k < MorselsPerAttack; ++k) {
        float phase = timestep / MorselTravelTimesteps + seed + float(k) / float(MorselsPerAttack);
        float progress = fract(phase);

        // The time step repeats after this number of cycles
        float cycle = mod(floor(phase), 1024.0);
        float eased = 0.6 * progress * progress * (3.0 - 2.0 * progress) + 0.4 * progress;

        // Each morsel takes its own path
        float side = (hash(vec2(cycle, seed * 97.0 + float(k))) - 0.5) * 0.5;
        vec2 control = 0.5 * (start + attacker) + normal * side * distance * 0.5;
        vec2 center = mix(mix(start, control, eased), mix(control, attacker, eased), eased);

        float radius = mix(MorselStartRadius, MorselEndRadius, eased) * zoom;
        float opacity = smoothstep(0.0, 0.12, progress) * (1.0 - smoothstep(0.82, 1.0, progress)) * intensity * smoothstep(0.6, 2.0, radius);
        if (opacity > 0.0) {
            emitMorsel(center, radius, opacity, light, fract(seed * 13.1 + cycle * 0.37 + float(k) * 0.29));
        }
    }
}
)";
}
