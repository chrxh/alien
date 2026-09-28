#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const DetonationEventFS = R"(
#version 330 core

// Explosion of a detonator cell, drawn over the merged scene with premultiplied alpha: rgb adds light,
// alpha removes scene light where smoke absorbs it or where the refracted scene replaces it.
// All sizes are measured in detonator radii. Applied techniques:
//
//   Blackbody color ramp              temperature driven fireball, sparks and light color
//   Domain warped fBm                 turbulent, billowing fireball and smoke
//   Screen space normals              relief lighting of surrounding bodies from the luminance gradient
//   Screen space ray marching         soft shadows and volumetric light shafts behind bodies
//   Blinn-Phong specular              glints on lit bodies
//   Refraction with dispersion        shock wave bending the scene with chromatic fringes
//   Heat haze                         shimmering air around the fireball
//   Starburst and anamorphic streak   lens effects of the initial flash

in vec2 localPos;
flat in vec2 centerPixelPos;
flat in float blastRadius;
flat in float progress;
flat in float seed;

out vec4 FragColor;

uniform sampler2D inputTexture1;
uniform vec2 viewportSize;
uniform float zoom;
uniform float time;

#define PI 3.1415926538
#define LIGHT_REACH 5.5
#define NUM_SHADOW_SAMPLES 12
#define NUM_SPARK_LAYERS 2

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

float fbm(vec2 pos)
{
    mat2 rotation = mat2(0.8, 0.6, -0.6, 0.8);
    float result = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 4; ++i) {
        result += amplitude * valueNoise(pos);
        pos = rotation * pos * 2.03;
        amplitude *= 0.5;
    }
    return result / 0.9375;
}

// 0.3 deep red, 0.6 orange, 0.85 yellow, 1.0 white, above bluish white
vec3 blackbody(float temperature)
{
    vec3 color = vec3(
        smoothstep(0.1, 0.45, temperature),
        pow(smoothstep(0.3, 0.85, temperature), 1.3) * 0.85,
        smoothstep(0.65, 1.15, temperature) * 0.75);
    color.b += smoothstep(1.0, 1.5, temperature) * 0.4;
    return color;
}

float luminance(vec3 color)
{
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float occupancy(vec3 color)
{
    return smoothstep(0.02, 0.25, luminance(color));
}

float distanceToSegment(vec2 pos, vec2 start, vec2 end)
{
    vec2 toPos = pos - start;
    vec2 segment = end - start;
    float fraction = clamp(dot(toPos, segment) / max(dot(segment, segment), 1e-6), 0.0, 1.0);
    return length(toPos - segment * fraction);
}

vec3 calcSparks(vec2 pos, float t, float pixelsPerRadius)
{
    vec3 result = vec3(0.0);
    float angle = atan(pos.y, pos.x) + PI;
    float width = max(0.03, 1.2 / pixelsPerRadius);

    // Each angular sector holds one spark flying radially outwards, so only the neighboring sectors need to be tested
    for (int layer = 0; layer < NUM_SPARK_LAYERS; ++layer) {
        float numSectors = layer == 0 ? 36.0 : 21.0;
        float sector = floor(angle / (2.0 * PI) * numSectors);
        for (int neighbor = -1; neighbor <= 1; ++neighbor) {
            float index = mod(sector + float(neighbor), numSectors);
            vec2 key = vec2(index, seed * 113.0 + float(layer) * 17.0);
            float speed = 1.5 + 3.5 * hash(key + 11.3);
            float lifetime = 0.35 + 0.55 * hash(key + 23.7);
            float sparkAngle = (index + hash(key)) / numSectors * 2.0 * PI - PI;
            vec2 sparkDir = vec2(cos(sparkAngle), sin(sparkAngle));

            float travel = 0.2 + speed * (1.0 - exp(-t * 3.0));
            float trail = (0.03 + 0.15 * exp(-t * 5.0)) * speed;
            vec2 head = sparkDir * travel;
            vec2 tail = sparkDir * max(travel - trail, 0.0);
            float sparkDist = distanceToSegment(pos, tail, head);

            float life = 1.0 - smoothstep(lifetime * 0.7, lifetime, t);
            float twinkle = 0.6 + 0.8 * hash(key + floor(time * 24.0));
            float temperature = mix(1.2, 0.35, clamp(t / lifetime, 0.0, 1.0));
            result += blackbody(temperature) * exp(-sparkDist * sparkDist / (width * width)) * life * twinkle * 5.0;
        }
    }
    return result;
}

void main()
{
    float t = progress;
    vec2 pos = localPos / blastRadius;
    float dist = length(pos);
    if (dist > LIGHT_REACH) {
        discard;
    }
    vec2 dir = dist > 1e-4 ? pos / dist : vec2(0.0);
    vec2 seedOffset = vec2(seed * 71.3, seed * 37.9);
    float flicker = valueNoise(vec2(time * 17.0, seed * 91.0));

    vec2 texel = 1.0 / viewportSize;
    vec2 uv = gl_FragCoord.xy * texel;
    float pixelsPerRadius = blastRadius * zoom;

    float flash = exp(-t * 14.0);
    float growth = 1.0 - exp(-t * 8.0);
    float cooling = exp(-t * 2.6);
    float ending = 1.0 - smoothstep(0.7, 1.0, t);

    // Shock wave and heat haze bend the scene
    float shockRadius = 3.4 * pow(1.0 - exp(-t * 5.0), 0.8);
    float shockWidth = 0.07 + 0.3 * t;
    float shockOffset = (dist - shockRadius) / shockWidth;
    float shockProfile = exp(-shockOffset * shockOffset);
    float shockFade = exp(-t * 3.2) * smoothstep(0.0, 0.03, t) * ending;

    vec2 hazeFlow = pos * 6.0 + vec2(0.0, -time * 3.0) + seedOffset;
    vec2 haze = vec2(valueNoise(hazeFlow), valueNoise(hazeFlow + vec2(5.2, 1.3))) - 0.5;
    float hazeStrength = exp(-dist * dist * 0.8) * smoothstep(0.05, 0.25, t) * ending;

    vec2 displacement = (-dir * shockOffset * shockProfile * shockFade * 0.45 + haze * hazeStrength * 0.06) * pixelsPerRadius;
    vec3 refractedScene = vec3(
        texture(inputTexture1, uv + displacement * 1.25 * texel).r,
        texture(inputTexture1, uv + displacement * texel).g,
        texture(inputTexture1, uv + displacement * 0.75 * texel).b);
    float refraction = clamp(length(displacement) * 0.25, 0.0, 1.0);

    // Fireball and smoke
    vec3 fire = vec3(0.0);
    float smoke = 0.0;
    vec3 smokeColor = vec3(0.0);
    if (dist < 3.2) {
        float fireballRadius = 0.2 + 1.2 * growth;
        vec2 flowPos = pos * 2.2 - dir * t * 2.0 + seedOffset;
        vec2 warp = vec2(fbm(flowPos * 0.7 + time * 0.4), fbm(flowPos * 0.7 - time * 0.35 + 3.1));
        float billow = fbm(flowPos + warp * 1.2);
        float lobes = fbm(dir * 2.2 + seedOffset * 1.7);
        float edge = fireballRadius * (0.65 + 0.7 * lobes);
        float fireDist = dist + (billow - 0.5) * 0.7 * fireballRadius;
        float fireMask = (1.0 - smoothstep(edge * 0.45, edge, fireDist)) * ending;
        float heat = clamp(1.0 - fireDist / edge, 0.0, 1.0);
        float temperature = (0.3 + 0.9 * heat) * (0.7 + 0.6 * billow) * cooling * (0.9 + 0.2 * flicker);
        fire = blackbody(temperature) * fireMask * (0.3 + 6.0 * temperature * temperature * temperature);

        float smokeRadius = 0.5 + 1.3 * growth + 0.5 * t;
        float smokeNoise = fbm(pos * 1.6 - dir * t * 0.9 + seedOffset * 2.3 + vec2(time * 0.07, 0.0));
        float smokeShape = 1.0 - smoothstep(smokeRadius * 0.4, smokeRadius * (0.8 + 0.4 * smokeNoise), dist);
        smoke = smokeShape * smoothstep(0.3, 0.7, smokeNoise) * smoothstep(0.1, 0.35, t) * (1.0 - smoothstep(0.55, 1.0, t)) * 0.7;
        smokeColor = vec3(0.04, 0.035, 0.03) + blackbody(0.55) * 0.5 * cooling * (1.0 - smoothstep(0.0, 1.5, dist));
    }

    // Lens effects of the flash
    vec3 core = vec3(1.0, 0.95, 0.88) * 25.0 * flash * exp(-dist * dist * 8.0);
    float rayPattern = pow(valueNoise(dir * 6.0 + seedOffset) * 0.55 + valueNoise(dir * 17.0 - seedOffset) * 0.45, 4.0);
    vec3 rays = vec3(1.0, 0.85, 0.6) * rayPattern * 10.0 * exp(-t * 5.0) / (1.0 + dist * dist * 1.2) * smoothstep(0.0, 0.3, dist);
    float streak = exp(-abs(pos.y) * 30.0) * exp(-abs(pos.x) * 0.7) * exp(-t * 8.0) * 4.0;
    vec3 flare = vec3(0.45, 0.7, 1.4) * streak;
    vec3 shockGlow = vec3(0.75, 0.85, 1.2) * shockProfile * shockFade * (0.5 + fbm(dir * 9.0 + seedOffset)) * 1.2;

    vec3 sparks = t < 0.95 ? calcSparks(pos, t, pixelsPerRadius) : vec3(0.0);

    // Illumination of the surroundings
    float lightStrength = (14.0 * flash + 3.2 * cooling * (0.75 + 0.5 * flicker)) * ending;
    vec3 lightColor = mix(blackbody(0.45 + 0.35 * cooling), vec3(0.95, 0.97, 1.1), flash);
    float reachFade = 1.0 - smoothstep(0.55, 1.0, dist / LIGHT_REACH);
    float attenuation = reachFade / (1.0 + dist * dist * 0.8);

    vec3 scene = texture(inputTexture1, uv).rgb;
    vec3 surfaceLight = vec3(0.0);
    vec3 airLight = vec3(0.0);
    if (lightStrength * attenuation > 0.005) {
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

        vec3 incident = lightColor * lightStrength * attenuation * shadow;
        surfaceLight = incident * (scene * (0.3 + 1.7 * diffuse) + occupancy(scene) * specular * 0.6) * (1.0 - smoke);

        float airDensity = 0.12 * reachFade / (1.0 + dist * dist * 1.5);
        airLight = lightColor * lightStrength * airDensity * mix(1.0, shadow, 0.85);
    }

    vec3 emission = core + rays + flare + shockGlow + fire * (1.0 - 0.5 * smoke) + sparks;
    float keptScene = (1.0 - refraction) * (1.0 - smoke);
    vec3 color = refractedScene * refraction * (1.0 - smoke) + smokeColor * smoke + emission + surfaceLight + airLight;
    FragColor = vec4(color, 1.0 - keptScene);
}
)";
}
