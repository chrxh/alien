#pragma once

#include <cstdint>
#include <filesystem>
#include <string>

#include <imgui.h>

#include <Base/Interface/Definitions.h>

#include "Definitions.h"
#include "PictureData.h"
#include "RenderView.h"
#include "TextureData.h"

// Renders the frames of the window: the simulation as background and the user interface on top.
// The implementation encapsulates the graphics API.
class _RenderingFacade
{
public:
    virtual ~_RenderingFacade() = default;

    static RenderingFacade get();

    //*********************
    //* Setup and shutdown
    //*********************

    // Prepares GLFW for the creation of the window
    virtual void setWindowHints() = 0;

    // Requires the ImGui context. The graphics device of the GPU engine is preferred, so that both can share memory.
    virtual void setup(GLFWwindow* window) = 0;

    // The font is used for the labels of the cell types
    virtual void setupSimulationRendering(ImFont* labelFont) = 0;

    // Requires that the simulation is closed since the GPU engine writes into memory of the renderer
    virtual void shutdown() = 0;

    //*********
    //* Frames
    //*********
    virtual void newFrame() = 0;

    // Shows the simulation as background of the current frame
    virtual void drawSimulation(RenderView const& view) = 0;

    // Fills the screen of the current frame with a color instead of the simulation
    virtual void clearScreen(FloatColorRGB const& color) = 0;

    virtual void render(ImDrawData* drawData) = 0;

    //***********
    //* Pictures
    //***********

    // Throws an AlienException if the picture cannot be rendered, e.g. because the GPU memory does not suffice
    virtual PictureData renderSimulationPicture(RenderView const& view) = 0;

    // Renders the draw list independently of the frames, e.g. for preview pictures
    virtual PictureData renderDrawList(ImDrawList* drawList, IntVector2D const& resolution, ImColor const& backgroundColor, int supersampling) = 0;

    //***********************************
    //* Textures for the user interface
    //***********************************
    virtual TextureData loadTexture(std::filesystem::path const& filename) = 0;
    virtual TextureData loadTextureFromMemory(std::string const& encodedImage) = 0;
    virtual TextureData createTexture(uint8_t const* pixels, int width, int height, TextureFormat format, TextureFilter filter) = 0;
    virtual void deleteTexture(TextureData const& texture) = 0;
    virtual void deleteTexture(ImTextureID textureId) = 0;

protected:
    static RenderingFacade _instance;
};
