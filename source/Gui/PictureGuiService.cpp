#include "PictureGuiService.h"

#include <algorithm>
#include <ranges>

#include <span>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <Base/AlienExceptions.h>
#include <Base/ExitScopeGuard.h>
#include <Base/LoggingService.h>

#include "PreviewDescView.h"
#include "SimulationView.h"
#include "StyleService.h"
#include "VulkanContext.h"
#include "VulkanFrameRenderer.h"
#include "WindowController.h"

namespace
{
    auto constexpr JpgQuality = 92;

    auto constexpr PreviewPictureResolution = PictureGuiService::PreviewPictureResolution;
    auto constexpr PreviewPictureBrightness = 1.3f;
    auto constexpr PreviewPictureSupersampling = 2;
}

std::optional<std::string> PictureGuiService::createSimulationPreviewJpg()
{
    try {
        auto screenWidth = WindowController::get().getWindowData().mode->width;
        auto scaleFactor = toFloat(screenWidth) / toFloat(PreviewPictureResolution.x);
        auto renderResolution = IntVector2D{screenWidth, toInt(toFloat(PreviewPictureResolution.y) * scaleFactor)};

        auto picture = SimulationView::get().savePicture(renderResolution);
        auto preview = scale(picture, PreviewPictureResolution);
        return encodeJpg(brighten(preview, PreviewPictureBrightness));
    } catch (AlienException const& exception) {
        log(Priority::Important, std::string("preview picture could not be created: ") + exception.what());
        return std::nullopt;
    }
}

namespace
{
    PictureData renderOffscreen(ImDrawList* drawList, IntVector2D const& resolution, ImColor const& backgroundColor, int supersampling)
    {
        IntVector2D renderResolution{resolution.x * supersampling, resolution.y * supersampling};

        // The user interface pipeline is made for the format of the screen
        auto& context = VulkanContext::get();
        auto format = VulkanFrameRenderer::get().getScreenFormat();
        auto image = context.createImage(renderResolution, format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        auto readbackBuffer = context.createBuffer(
            static_cast<VkDeviceSize>(renderResolution.x) * renderResolution.y * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VulkanMemory::HostVisible);
        ExitScopeGuard releaseResources([&] {
            context.destroyImage(image);
            context.destroyBuffer(readbackBuffer);
        });

        ImDrawData drawData;
        drawData.Valid = true;
        drawData.DisplayPos = {0, 0};
        drawData.DisplaySize = {toFloat(resolution.x), toFloat(resolution.y)};
        drawData.FramebufferScale = {toFloat(supersampling), toFloat(supersampling)};
        drawData.AddDrawList(drawList);

        context.submitAndWait([&](VkCommandBuffer commandBuffer) {
            VulkanContext::useImage(commandBuffer, image, ImageUsage::ColorAttachment);
            VkRenderingAttachmentInfo colorAttachment{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .imageView = image.view,
                .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                .clearValue = {.color = {.float32 = {backgroundColor.Value.x, backgroundColor.Value.y, backgroundColor.Value.z, 1.0f}}},
            };
            VkRenderingInfo renderingInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = {{0, 0}, {static_cast<uint32_t>(renderResolution.x), static_cast<uint32_t>(renderResolution.y)}},
                .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &colorAttachment,
            };
            vkCmdBeginRendering(commandBuffer, &renderingInfo);
            ImGui_ImplVulkan_RenderDrawData(&drawData, commandBuffer);
            vkCmdEndRendering(commandBuffer);

            VulkanContext::useImage(commandBuffer, image, ImageUsage::TransferSource);
            VkBufferImageCopy region{
                .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                .imageExtent = {static_cast<uint32_t>(renderResolution.x), static_cast<uint32_t>(renderResolution.y), 1},
            };
            vkCmdCopyImageToBuffer(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuffer.buffer, 1, &region);
        });

        // Unlike with OpenGL, the rows are already ordered from top to bottom
        PictureData result{
            .resolution = renderResolution,
            .pixels = std::vector<uint8_t>(static_cast<size_t>(renderResolution.x) * renderResolution.y * PictureData::NumChannels)};
        auto isBgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
        auto pixels = std::span(static_cast<uint8_t const*>(readbackBuffer.mapped), static_cast<size_t>(renderResolution.x) * renderResolution.y * 4);
        for (auto const& [source, destination] : std::views::zip(pixels | std::views::chunk(4), result.pixels | std::views::chunk(PictureData::NumChannels))) {
            destination[0] = isBgra ? source[2] : source[0];
            destination[1] = source[1];
            destination[2] = isBgra ? source[0] : source[2];
        }
        return result;
    }
}

std::optional<std::string> PictureGuiService::createGenomePreviewJpg(std::vector<PreviewDesc> const& previews)
{
    if (previews.empty()) {
        return std::nullopt;
    }
    try {
        RealVector2D viewSize{toFloat(PreviewPictureResolution.x), toFloat(PreviewPictureResolution.y)};

        // The collage is built in an own draw list so that it can be rendered independently of the current ImGui frame
        ImDrawList drawList(ImGui::GetDrawListSharedData());
        drawList._ResetForNewFrame();
        drawList.PushTextureID(ImGui::GetIO().Fonts->TexID);
        drawList.PushClipRect({0, 0}, {viewSize.x, viewSize.y}, false);

        auto previewView = _PreviewDescView::create();
        previewView->drawCollage(&drawList, previews, {0, 0}, viewSize);

        drawList.PopClipRect();
        drawList.PopTextureID();

        auto picture = renderOffscreen(&drawList, PreviewPictureResolution, Const::GenomePreviewBackgroundColor, PreviewPictureSupersampling);
        return encodeJpg(brighten(scale(picture, PreviewPictureResolution), PreviewPictureBrightness));
    } catch (AlienException const& exception) {
        log(Priority::Important, std::string("preview picture could not be created: ") + exception.what());
        return std::nullopt;
    }
}

PictureData PictureGuiService::scale(PictureData const& picture, IntVector2D const& resolution)
{
    if (resolution.x <= 0 || resolution.y <= 0) {
        throw AlienException("The resolution of a picture must be positive.");
    }

    PictureData result{.resolution = resolution, .pixels = std::vector<uint8_t>(static_cast<size_t>(resolution.x) * resolution.y * PictureData::NumChannels)};
    auto scaled = stbir_resize_uint8_srgb(
        picture.pixels.data(),
        picture.resolution.x,
        picture.resolution.y,
        picture.resolution.x * PictureData::NumChannels,
        result.pixels.data(),
        resolution.x,
        resolution.y,
        resolution.x * PictureData::NumChannels,
        STBIR_RGB);
    if (scaled == nullptr) {
        throw AlienException("The picture could not be scaled.");
    }
    return result;
}

PictureData PictureGuiService::brighten(PictureData const& picture, float factor)
{
    auto result = picture;
    for (auto& value : result.pixels) {
        value = static_cast<uint8_t>(std::min(255.0f, toFloat(value) * factor));
    }
    return result;
}

std::string PictureGuiService::encodeJpg(PictureData const& picture)
{
    std::string result;
    auto appendData = [](void* context, void* data, int size) { static_cast<std::string*>(context)->append(static_cast<char const*>(data), size); };
    auto writeResult =
        stbi_write_jpg_to_func(appendData, &result, picture.resolution.x, picture.resolution.y, PictureData::NumChannels, picture.pixels.data(), JpgQuality);
    if (writeResult == 0) {
        throw AlienException("The picture could not be encoded.");
    }
    return result;
}

std::string PictureGuiService::encodePng(PictureData const& picture)
{
    std::string result;
    auto appendData = [](void* context, void* data, int size) { static_cast<std::string*>(context)->append(static_cast<char const*>(data), size); };
    auto writeResult = stbi_write_png_to_func(
        appendData,
        &result,
        picture.resolution.x,
        picture.resolution.y,
        PictureData::NumChannels,
        picture.pixels.data(),
        picture.resolution.x * PictureData::NumChannels);
    if (writeResult == 0) {
        throw AlienException("The picture could not be encoded.");
    }
    return result;
}

void PictureGuiService::savePng(PictureData const& picture, std::filesystem::path const& filename)
{
    auto result = stbi_write_png(
        filename.string().c_str(),
        picture.resolution.x,
        picture.resolution.y,
        PictureData::NumChannels,
        picture.pixels.data(),
        picture.resolution.x * PictureData::NumChannels);
    if (result == 0) {
        throw AlienException("The file could not be written.");
    }
}
