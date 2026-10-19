#pragma once

#include <vector>

#include <Base/MathTypes.h>

#include <Data/Definitions.h>

#include <Engine/Interface/GeometryBuffers.h>

#include <Rendering/Interface/RenderView.h>

// Grids and lights of a frame of the ray traced view, which follow from the view and the parameters alone. They are determined before the
// engine extracts the objects, since the engine collects the objects outside the extracted region in the shadow grid.
struct RayTracingLayout
{
    // The resources of the ray tracing are allocated for the largest grids
    static constexpr int MaxGasNodes = 512;
    static constexpr int MaxLightClusters = 256;
    static constexpr int GasIrradianceImageSize = MaxGasNodes;
    static constexpr int MaxShadowGridNodes = 512;

    // The light of the cells falls off beyond this distance close up, see the gas grid
    static constexpr float LightFalloffDistance = 5.0f;

    // The light reaches a position from the part of the area of a source next to it. Its half width grows with this factor times the distance
    // to the area, so that the shadows become softer the farther they fall.
    static constexpr float SourceLightSpread = 0.25f;

    // Grid in world coordinates whose nodes hold the density of the gas. The nodes are aligned to the world and their distance is a power of
    // two multiple of a base distance, so that moving or zooming the view keeps the gas stable.
    struct GasGrid
    {
        RealVector2D origin;
        float nodeDistance = 1.0f;
        IntVector2D size;
        int blurRadius = 0;
        float cloudSpreadInNodes = 1.0f;

        // The light of the cells is collected in clusters and reaches the nodes of the irradiance grid
        float lightFalloffDistance = 1.0f;
        float lightRange = 1.0f;
        RealVector2D lightClusterOrigin;
        float lightClusterSize = 1.0f;
        IntVector2D numLightClusters;
        float irradianceNodeDistance = 1.0f;
        IntVector2D irradianceSize;
    };

    // A radiation source that shines from its circular or rectangular area
    struct SourceLight
    {
        RealVector2D center;
        RealVector2D halfExtent;
        float range = 0.0f;
        bool circular = true;
        FloatColorRGB color;
    };

    GasGrid gasGrid;
    std::vector<SourceLight> sourceLights;
    ShadowGridLayout shadowGrid;

    // A view with an infinite or undefined extent results in an empty layout
    static RayTracingLayout calc(RenderView const& view, SimulationParameters const& parameters, IntVector2D const& worldSize);
    bool isEmpty() const;
};
