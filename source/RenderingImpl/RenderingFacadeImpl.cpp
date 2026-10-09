#include "RenderingFacadeImpl.h"

#include <ranges>
#include <span>

#include <GLFW/glfw3.h>

#include <imgui_impl_vulkan.h>

#include <Base/ExitScopeGuard.h>
#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>

#include <EngineInterface/SimulationFacade.h>

#include "SimulationRenderer.h"
#include "TextureService.h"
#include "VulkanContext.h"
#include "VulkanFrameRenderer.h"
#include "VulkanGeometryBuffers.h"

void _RenderingFacadeImpl::set(RenderingFacade const& instance)
{
    _instance = instance;
}

void _RenderingFacadeImpl::setWindowHints()
{
    // Vulkan renders into a surface of the window instead of an OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
}

namespace
{
    // Errors of the GPU engine are reported when the simulation is created
    std::optional<GpuUuid> getEngineGpuUuid()
    {
        try {
            return _SimulationFacade::get()->getGpuUuid();
        } catch (std::exception const&) {
            return std::nullopt;
        }
    }

    // Geometry buffers created afterwards are only shareable with the GPU engine if the check succeeds
    void checkInterop()
    {
        if (!GlobalSettings::get().isInterop() || !VulkanContext::get().isMemorySharingSupported()) {
            return;
        }
        auto interopWorking = false;
        try {
            // Without objects, the buffers get their minimum capacity
            auto geometryBuffers = _VulkanGeometryBuffers::create();
            geometryBuffers->updateNumObjects({});
            interopWorking = _SimulationFacade::get()->isRenderingInteropWorking(geometryBuffers);
        } catch (std::exception const& exception) {
            log(Priority::Important, std::string("CUDA-Vulkan interop check failed: ") + exception.what());
        }
        if (interopWorking) {
            log(Priority::Important, "CUDA-Vulkan interop is working");
        } else {
            GlobalSettings::get().setInterop(false);
            log(Priority::Important, "CUDA-Vulkan interop is not working on this system, falling back to the transfer over host memory");
        }
    }
}

void _RenderingFacadeImpl::setup(GLFWwindow* window)
{
    VulkanContext::get().setup(window, getEngineGpuUuid());
    checkInterop();
    VulkanFrameRenderer::get().setup(window);
}

void _RenderingFacadeImpl::setupSimulationRendering(ImFont* labelFont)
{
    SimulationRenderer::get().setup(labelFont);
}

void _RenderingFacadeImpl::shutdown()
{
    SimulationRenderer::get().shutdown();

    auto& context = VulkanContext::get();
    context.releasePendingResources();
    TextureService::get().shutdown();
    VulkanFrameRenderer::get().shutdown();
    context.shutdown();
}

void _RenderingFacadeImpl::newFrame()
{
    VulkanFrameRenderer::get().newFrame();
}

void _RenderingFacadeImpl::drawSimulation(RenderView const& view)
{
    SimulationRenderer::get().draw(view);
}

void _RenderingFacadeImpl::clearScreen(FloatColorRGB const& color)
{
    VulkanFrameRenderer::get().clearScreen(color);
}

void _RenderingFacadeImpl::render(ImDrawData* drawData)
{
    VulkanFrameRenderer::get().render(drawData);
}

PictureData _RenderingFacadeImpl::renderSimulationPicture(RenderView const& view)
{
    return SimulationRenderer::get().renderPicture(view);
}

PictureData _RenderingFacadeImpl::renderDrawList(ImDrawList* drawList, IntVector2D const& resolution, ImColor const& backgroundColor, int supersampling)
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

TextureData _RenderingFacadeImpl::loadTexture(std::filesystem::path const& filename)
{
    return TextureService::get().loadTexture(filename);
}

TextureData _RenderingFacadeImpl::loadTextureFromMemory(std::string const& encodedImage)
{
    return TextureService::get().loadTextureFromMemory(encodedImage);
}

TextureData _RenderingFacadeImpl::createTexture(uint8_t const* pixels, int width, int height, TextureFormat format, TextureFilter filter)
{
    return TextureService::get().createTexture(pixels, width, height, format, filter);
}

void _RenderingFacadeImpl::deleteTexture(TextureData const& texture)
{
    TextureService::get().deleteTexture(texture);
}

void _RenderingFacadeImpl::deleteTexture(ImTextureID textureId)
{
    TextureService::get().deleteTexture(textureId);
}
