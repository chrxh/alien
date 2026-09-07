#pragma once

#include <string_view>

namespace Shaders
{
    std::string_view const NonFluidObjectFS = R"(
#version 330 core
out vec4 FragColor;

in vec3 vColor;

uniform float brightness;

void main()
{
    // Coordinates on the sprite in [-1, 1]
    vec2 coord = (gl_PointCoord - vec2(0.5, 0.5)) * 2.0;
    float radiusSquared = dot(coord, coord);
    if (radiusSquared > 1.0) {
        discard;
    }

    // Shade the sprite as a translucent vesicle rather than a solid sphere: a wide diffuse term
    // for the volume and a rim that lights up where the body is thin. A narrow highlight would
    // make the cell look like polished plastic.
    // The y axis of the sprite points downwards, hence the sign of the normal.
    vec3 normal = vec3(coord.x, -coord.y, sqrt(1.0 - radiusSquared));
    vec3 lightDirection = normalize(vec3(-0.55, 0.55, 0.45));
    vec3 halfVector = normalize(lightDirection + vec3(0.0, 0.0, 1.0));

    float diffuse = 0.55 + 0.45 * (0.5 + 0.5 * dot(normal, lightDirection));
    float translucentRim = 0.3 * radiusSquared * radiusSquared;
    float sheen = 0.12 * pow(max(dot(normal, halfVector), 0.0), 8.0);
    float silhouette = 1.0 - smoothstep(0.7, 1.0, radiusSquared);

    // The metaballs step derives the silhouette of a cell from its brightness, so dimming here also shrinks
    // the cell. The caller therefore keeps the brightness at 1 while the cells are meant to fuse with the
    // bodies and dims them only where that does not matter.
    FragColor = vec4(vColor * (diffuse + translucentRim + sheen) * silhouette * brightness, 1.0);
}
)";
}
