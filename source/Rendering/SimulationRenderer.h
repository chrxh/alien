#pragma once

#include <Base/Singleton.h>

#include "Definitions.h"
#include "PictureData.h"
#include "RenderView.h"

// Renders the simulation into the frames of the window and into pictures
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
    void createRenderGraph(ImFont* labelFont);
    PictureData renderPictureInternal(RenderView const& view);

    RenderGraph _renderGraph;
};
