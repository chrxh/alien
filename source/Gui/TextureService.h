#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>

#include <imgui.h>

#include <Base/Singleton.h>

#include "Definitions.h"
#include "VulkanContext.h"

enum class TextureFormat
{
    Rgba,
    Bgra,
};

enum class TextureFilter
{
    Smooth,
    Nearest,
};

// Textures for the user interface
class TextureService
{
    MAKE_SINGLETON(TextureService);

public:
    void shutdown();

    TextureData loadTexture(std::filesystem::path const& filename);
    TextureData loadTextureFromMemory(std::string const& encodedImage);
    TextureData
    createTexture(uint8_t const* pixels, int width, int height, TextureFormat format = TextureFormat::Rgba, TextureFilter filter = TextureFilter::Smooth);

    void deleteTexture(TextureData const& texture);
    void deleteTexture(ImTextureID textureId);

private:
    VkSampler getSampler(TextureFilter filter);

    struct Texture
    {
        VulkanImage image;
        VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    };
    std::unordered_map<ImTextureID, Texture> _textures;

    VkSampler _smoothSampler = VK_NULL_HANDLE;
    VkSampler _nearestSampler = VK_NULL_HANDLE;
};
