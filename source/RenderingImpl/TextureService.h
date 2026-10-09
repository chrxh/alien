#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include <imgui.h>

#include <Base/Singleton.h>

#include <RenderingInterface/TextureData.h>

#include "Definitions.h"

// Textures for the user interface
class TextureService
{
    MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTION(TextureService);

public:
    ~TextureService();

    void shutdown();

    TextureData loadTexture(std::filesystem::path const& filename);
    TextureData loadTextureFromMemory(std::string const& encodedImage);
    TextureData
    createTexture(uint8_t const* pixels, int width, int height, TextureFormat format = TextureFormat::Rgba, TextureFilter filter = TextureFilter::Smooth);

    void deleteTexture(TextureData const& texture);
    void deleteTexture(ImTextureID textureId);

private:
    TextureService();

    struct Resources;
    std::unique_ptr<Resources> _resources;
};
