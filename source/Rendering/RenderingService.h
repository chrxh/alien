#pragma once

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include "Definitions.h"
#include "PictureData.h"

// Renders the frames of the window: the simulation as background and the user interface on top
class RenderingService
{
    MAKE_SINGLETON(RenderingService);

public:
    // Requires the ImGui context. The graphics device of the GPU engine is preferred, so that both can share memory.
    void setup(GLFWwindow* window);

    // Requires that the simulation renderer is already shut down
    void shutdown();

    void newFrame();

    // Fills the screen of the current frame with a color instead of the simulation
    void clearScreen(FloatColorRGB const& color);

    void render(ImDrawData* drawData);

    // Renders the draw list independently of the frames, e.g. for preview pictures
    PictureData renderDrawList(ImDrawList* drawList, IntVector2D const& resolution, ImColor const& backgroundColor, int supersampling);
};
