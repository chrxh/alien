#pragma once

#include <imgui.h>

#include <Base/Singleton.h>

#include <Rendering/Interface/PictureData.h>
#include <Rendering/Interface/RenderView.h>

#include "Definitions.h"
#include "VulkanGeometryBuffers.h"

// Renders the simulation into the frames of the window and into pictures, either by rasterization or with hardware ray tracing
class SimulationRenderer
{
    MAKE_SINGLETON(SimulationRenderer);

public:
    // The font is used for the labels of the cell types
    void setup(ImFont* labelFont);
    void shutdown();

    bool isRayTracingSupported() const;

    // Shows the simulation as background of the current frame
    void draw(RenderView const& view);

    // Throws an AlienException if the picture cannot be rendered, e.g. because the GPU memory does not suffice
    PictureData renderPicture(RenderView const& view);

private:
    void createRasterGraph(ImFont* labelFont);
    void createRayTracingGraph(ImFont* labelFont);
    PictureData renderPictureInternal(RenderView const& view);

    void updateGeometry(RealRect const& visibleWorldRect);

    // Ray tracing falls back to rasterization if its resources cannot be allocated, frames only retry when it is requested anew
    RenderGraph const& selectGraph(RenderView const& view, bool forPicture);

    VulkanGeometryBuffers _geometryBuffers;
    RenderGraph _rasterGraph;
    RenderGraph _rayTracingGraph;
    RayTracingRenderStep _rayTracingStep;
    bool _rayTracingFailed = false;
};
