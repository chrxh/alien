#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const FluidParticleVS = R"(
#version 450
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aGlow;
layout (location = 3) in vec2 aVel;

out vec3 vColor;
out vec2 vAxis;
out float vElongation;

uniform vec2 worldSize;
uniform vec2 rectUpperLeft;
uniform float zoom;
uniform float radius;
uniform vec2 viewportSize;
uniform bool onBackground;
uniform float renderScale;

// The ray tracing renders the fluids itself as gas, only the cores of glowing fluids stay. The z coordinate of fluids is 1.
uniform bool energyParticlesOnly;

const float CoreHiddenZoom = 2.0;
const float CoreDetailZoom = 7.0;

// Weak energy particles contribute to the clouds with a softly raised minimum intensity, so that crowds of them merge into visible clouds. A
// higher minimum would let a few particles passing each other flash up as a cloud.
const float MinEnergyCloudIntensity = 0.2;

// The contributions of energy particles to the clouds stretch along their way into ellipses of the same area, the faster the longer. The
// cores and points of a few pixels stay round.
const float ElongationPerSpeed = 2.5;
const float MaxElongation = 5.0;
const float MinElongatedSize = 3.0;

void main()
{
    vAxis = vec2(1.0, 0.0);
    vElongation = 1.0;
    if (energyParticlesOnly && aPos.z >= 0.5 && (onBackground || aGlow <= 0.0)) {
        gl_Position = vec4(-2.0, -2.0, -2.0, 1.0);
        gl_PointSize = 0.0;
        vColor = vec3(0.0);
        return;
    }

    // Transform world position to normalized device coordinates
    vec2 relativePos = aPos.xy - rectUpperLeft;
    vec2 screenPos = relativePos * zoom;
    vec2 ndc = (screenPos / viewportSize) * 2.0 - 1.0;
    ndc.y = -ndc.y; // Flip Y coordinate
    gl_Position = vec4(ndc, 0.0, 1.0);

    bool energyParticle = aPos.z < 0.5;
    float ballSize;
    float coreVisibility = 1.0;
    if (onBackground) {
        ballSize = 15.0;
    } else {
        float screenZoom = zoom / renderScale;
        ballSize = screenZoom < CoreDetailZoom ? 0.0 : 0.2;
        coreVisibility = clamp((screenZoom - CoreHiddenZoom) / (CoreDetailZoom - CoreHiddenZoom), 0.0, 1.0);
    }

    float g = clamp(aGlow, 0.0, 1.0);
    if (onBackground) {
        vColor = aColor * mix(1.0, 4.0, g);
    } else {
        ballSize = mix(ballSize, 1.0, g);
        vColor = aColor * mix(coreVisibility, 20.0, g);
    }

    float size = radius * ballSize;
    if (energyParticle && onBackground) {

        // The blue component of an energy particle is its intensity, see Engine/Impl/Kernels/GeometryKernels.cu
        float intensity = max(aColor.b, 1e-4);
        vColor *= sqrt(intensity * intensity + MinEnergyCloudIntensity * MinEnergyCloudIntensity) / intensity;

        // The rows of the render targets run from the bottom to the top, hence the point coordinates run against the world y axis
        float speed = length(aVel);
        if (speed > 1e-4) {
            vAxis = vec2(aVel.x, -aVel.y) / speed;
            vElongation = mix(1.0, min(1.0 + ElongationPerSpeed * speed, MaxElongation), smoothstep(1.0, MinElongatedSize, size));
            size *= sqrt(vElongation);
        }
    }
    gl_PointSize = size;
}
)";
}
