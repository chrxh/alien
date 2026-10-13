#pragma once

#include <string_view>

namespace Shaders
{
    // Frames on the screen are smoothed within themselves, so that nothing changes over time when the simulation stops. Still pictures
    // average the frames of jittered rays instead.
    std::string_view const RayTracingAntiAliasingCS = R"(
const float FxaaSpanMax = 8.0;
const float FxaaReduceMultiplier = 1.0 / 8.0;
const float FxaaReduceMinimum = 1.0 / 128.0;

// Bright pixels would dominate the filter otherwise
vec3 compress(vec3 color)
{
    return color / (1.0 + luminance(color));
}

vec3 decompress(vec3 color)
{
    return color / max(1.0 - luminance(color), 0.01);
}

vec3 sampleCompressed(vec2 pixelPosition)
{
    return compress(texture(sceneColorSampler, pixelPosition / frame.view.xy).rgb);
}

// Fast approximate anti-aliasing by Timothy Lottes: blurs along the edges that the luminance reveals
vec3 applyFxaa(ivec2 pixel)
{
    vec2 center = vec2(pixel) + 0.5;
    float lumaM = luminance(sampleCompressed(center));
    float lumaNW = luminance(sampleCompressed(center + vec2(-1.0, -1.0)));
    float lumaNE = luminance(sampleCompressed(center + vec2(1.0, -1.0)));
    float lumaSW = luminance(sampleCompressed(center + vec2(-1.0, 1.0)));
    float lumaSE = luminance(sampleCompressed(center + vec2(1.0, 1.0)));
    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 direction = vec2(-((lumaNW + lumaNE) - (lumaSW + lumaSE)), (lumaNW + lumaSW) - (lumaNE + lumaSE));
    float directionReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * 0.25 * FxaaReduceMultiplier, FxaaReduceMinimum);
    direction = clamp(direction / (min(abs(direction.x), abs(direction.y)) + directionReduce), vec2(-FxaaSpanMax), vec2(FxaaSpanMax));

    vec3 colorA = 0.5 * (sampleCompressed(center + direction * (1.0 / 3.0 - 0.5)) + sampleCompressed(center + direction * (2.0 / 3.0 - 0.5)));
    vec3 colorB = colorA * 0.5 + 0.25 * (sampleCompressed(center - direction * 0.5) + sampleCompressed(center + direction * 0.5));
    float lumaB = luminance(colorB);
    return decompress(lumaB < lumaMin || lumaB > lumaMax ? colorA : colorB);
}

void main()
{
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (!isInView(pixel)) {
        return;
    }
    vec4 sceneColorValue = imageLoad(sceneColor, pixel);
    vec4 color = vec4(frame.parameters.y > 0.5 ? applyFxaa(pixel) : sceneColorValue.rgb, sceneColorValue.a);
    if (frame.parameters.z < 1.0) {
        color = mix(texelFetch(previousAntiAliased, pixel, 0), color, frame.parameters.z);
    }
    imageStore(antiAliased, pixel, color);
}
)";
}
