#include "TextureService.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <ranges>
#include <stdexcept>

#include <stb_image.h>
#include <imgui_impl_vulkan.h>

void TextureService::shutdown()
{
    auto& context = VulkanContext::get();
    context.waitIdle();
    for (auto& texture : _textures | std::views::values) {
        ImGui_ImplVulkan_RemoveTexture(texture.descriptorSet);
        context.destroyImage(texture.image);
    }
    _textures.clear();

    auto device = context.getDevice();
    vkDestroySampler(device, _smoothSampler, nullptr);
    vkDestroySampler(device, _nearestSampler, nullptr);
    _smoothSampler = VK_NULL_HANDLE;
    _nearestSampler = VK_NULL_HANDLE;
}

TextureData TextureService::loadTexture(std::filesystem::path const& filename)
{
    int width, height, numChannels;
    auto pixels = stbi_load(filename.string().c_str(), &width, &height, &numChannels, 4);
    if (pixels == nullptr) {
        throw std::runtime_error("Failed to load texture");
    }
    auto result = createTexture(pixels, width, height);
    stbi_image_free(pixels);
    return result;
}

TextureData TextureService::loadTextureFromMemory(std::string const& encodedImage)
{
    int width, height, numChannels;
    auto pixels =
        stbi_load_from_memory(reinterpret_cast<stbi_uc const*>(encodedImage.data()), static_cast<int>(encodedImage.size()), &width, &height, &numChannels, 4);
    if (pixels == nullptr) {
        throw std::runtime_error("Failed to load texture");
    }
    auto result = createTexture(pixels, width, height);
    stbi_image_free(pixels);
    return result;
}

namespace
{
    void transitionMipLevel(VkCommandBuffer commandBuffer, VkImage image, uint32_t level, VkImageLayout oldLayout, VkImageLayout newLayout)
    {
        VkImageMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1},
        };
        VkDependencyInfo dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }
}

TextureData TextureService::createTexture(uint8_t const* pixels, int width, int height, TextureFormat format, TextureFilter filter)
{
    auto& context = VulkanContext::get();
    auto sizeInBytes = static_cast<VkDeviceSize>(width) * height * 4;
    auto stagingBuffer = context.createBuffer(sizeInBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanMemory::HostVisible);
    std::memcpy(stagingBuffer.mapped, pixels, sizeInBytes);

    auto mipLevels = filter == TextureFilter::Smooth ? static_cast<uint32_t>(std::bit_width(static_cast<uint32_t>(std::max(width, height)))) : 1u;
    auto vkFormat = format == TextureFormat::Bgra ? VK_FORMAT_B8G8R8A8_UNORM : VK_FORMAT_R8G8B8A8_UNORM;
    auto image = context.createImage(
        {width, height}, vkFormat, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, mipLevels);

    context.submitAndWait([&](VkCommandBuffer commandBuffer) {
        VulkanContext::useImage(commandBuffer, image, ImageUsage::TransferDestination);
        VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1},
        };
        vkCmdCopyBufferToImage(commandBuffer, stagingBuffer.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        auto levelWidth = width;
        auto levelHeight = height;
        for (uint32_t level = 1; level < mipLevels; ++level) {
            transitionMipLevel(commandBuffer, image.image, level - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            auto nextWidth = std::max(1, levelWidth / 2);
            auto nextHeight = std::max(1, levelHeight / 2);
            VkImageBlit blit{
                .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 1},
                .srcOffsets = {{0, 0, 0}, {levelWidth, levelHeight, 1}},
                .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                .dstOffsets = {{0, 0, 0}, {nextWidth, nextHeight, 1}},
            };
            vkCmdBlitImage(
                commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            transitionMipLevel(commandBuffer, image.image, level - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            levelWidth = nextWidth;
            levelHeight = nextHeight;
        }
        transitionMipLevel(commandBuffer, image.image, mipLevels - 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    });
    image.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    context.destroyBuffer(stagingBuffer);

    auto descriptorSet = ImGui_ImplVulkan_AddTexture(getSampler(filter), image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    auto textureId = reinterpret_cast<ImTextureID>(descriptorSet);
    _textures.emplace(textureId, Texture{.image = image, .descriptorSet = descriptorSet});
    return {textureId, width, height};
}

void TextureService::deleteTexture(TextureData const& texture)
{
    deleteTexture(texture.textureId);
}

void TextureService::deleteTexture(ImTextureID textureId)
{
    auto findResult = _textures.find(textureId);
    if (findResult == _textures.end()) {
        return;
    }
    auto texture = findResult->second;
    _textures.erase(findResult);

    // The texture may still be part of the frame being prepared
    VulkanContext::get().destroyLater([texture]() mutable {
        ImGui_ImplVulkan_RemoveTexture(texture.descriptorSet);
        VulkanContext::get().destroyImage(texture.image);
    });
}

VkSampler TextureService::getSampler(TextureFilter filter)
{
    auto& sampler = filter == TextureFilter::Smooth ? _smoothSampler : _nearestSampler;
    if (sampler == VK_NULL_HANDLE) {
        auto vkFilter = filter == TextureFilter::Smooth ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        VkSamplerCreateInfo samplerInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = vkFilter,
            .minFilter = vkFilter,
            .mipmapMode = filter == TextureFilter::Smooth ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod = VK_LOD_CLAMP_NONE,
        };
        checkVkResult(vkCreateSampler(VulkanContext::get().getDevice(), &samplerInfo, nullptr, &sampler), "vkCreateSampler");
    }
    return sampler;
}
