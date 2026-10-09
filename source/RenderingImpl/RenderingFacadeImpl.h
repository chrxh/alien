#pragma once

#include <RenderingInterface/RenderingFacade.h>

#include "Definitions.h"

// Rendering with Vulkan
class _RenderingFacadeImpl : public _RenderingFacade
{
public:
    static void set(RenderingFacade const& instance);

    void setWindowHints() override;
    void setup(GLFWwindow* window) override;
    void setupSimulationRendering(ImFont* labelFont) override;
    void shutdown() override;

    void newFrame() override;
    void drawSimulation(RenderView const& view) override;
    void clearScreen(FloatColorRGB const& color) override;
    void render(ImDrawData* drawData) override;

    PictureData renderSimulationPicture(RenderView const& view) override;
    PictureData renderDrawList(ImDrawList* drawList, IntVector2D const& resolution, ImColor const& backgroundColor, int supersampling) override;

    TextureData loadTexture(std::filesystem::path const& filename) override;
    TextureData loadTextureFromMemory(std::string const& encodedImage) override;
    TextureData createTexture(uint8_t const* pixels, int width, int height, TextureFormat format, TextureFilter filter) override;
    void deleteTexture(TextureData const& texture) override;
    void deleteTexture(ImTextureID textureId) override;
};
