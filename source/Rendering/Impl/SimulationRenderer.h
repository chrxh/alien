#pragma once

#include <optional>

#include <imgui.h>

#include <Base/Singleton.h>

#include <Rendering/Interface/PictureData.h>
#include <Rendering/Interface/RenderView.h>

#include "Definitions.h"
#include "VulkanGeometryBuffers.h"

// Renders the simulation into the frames of the window and into pictures, with hardware ray tracing if the GPU supports it and by
// rasterization otherwise
class SimulationRenderer
{
    MAKE_SINGLETON(SimulationRenderer);

public:
    // The font is used for the labels of the cell types
    void setup(ImFont* labelFont);
    void shutdown();

    // Shows the simulation as background of the current frame
    void draw(RenderView const& view);

    // Throws an AlienException if the picture cannot be rendered, e.g. because the GPU memory does not suffice
    PictureData renderPicture(RenderView const& view);

private:
    void createRasterGraph(ImFont* labelFont);
    void createRayTracingGraph(ImFont* labelFont);
    PictureData renderPictureInternal(RenderView const& view);

    void updateGeometry(RealRect const& visibleWorldRect);

    // Ray tracing falls back to rasterization if its resources cannot be allocated. Frames only retry once the view size has changed or the
    // scene has clearly fewer objects, so that a lack of memory does not stall every frame.
    RenderGraph const& selectGraph(RenderView const& view, bool forPicture);

    struct RayTracingDemand
    {
        IntVector2D viewSize;
        uint64_t numSceneObjects = 0;
    };

    VulkanGeometryBuffers _geometryBuffers;
    RenderGraph _rasterGraph;
    RenderGraph _rayTracingGraph;
    RayTracingResources _rayTracingResources;
    std::optional<RayTracingDemand> _failedRayTracingDemand;
};
