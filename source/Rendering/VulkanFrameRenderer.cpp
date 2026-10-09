#include "VulkanFrameRenderer.h"

#include <algorithm>
#include <stdexcept>

#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

#include <Base/LoggingService.h>

#include "Shader.h"

namespace
{
    auto constexpr ImGuiDescriptorPoolSize = 8192;

    auto constexpr PresentationVS = R"(
#version 450
out vec2 texCoord;

void main()
{
    vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    texCoord = vec2(position.x, 1.0 - position.y);
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)";

    auto constexpr PresentationFS = R"(
#version 450
in vec2 texCoord;
out vec4 FragColor;

uniform sampler2D sceneTexture;

void main()
{
    FragColor = vec4(texture(sceneTexture, texCoord).rgb, 1.0);
}
)";

    void checkImGuiVkResult(VkResult result)
    {
        if (result < 0) {
            throw std::runtime_error("Vulkan error " + std::to_string(result) + " in the user interface backend.");
        }
    }
}

void VulkanFrameRenderer::setup(GLFWwindow* window)
{
    _window = window;
    auto& context = VulkanContext::get();
    auto device = context.getDevice();

    VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = context.getQueueFamily(),
    };
    checkVkResult(vkCreateCommandPool(device, &poolInfo, nullptr, &_commandPool), "vkCreateCommandPool");
    VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = _commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    checkVkResult(vkAllocateCommandBuffers(device, &allocateInfo, &_commandBuffer), "vkAllocateCommandBuffers");

    VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    checkVkResult(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &_imageAcquiredSemaphore), "vkCreateSemaphore");
    VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    checkVkResult(vkCreateFence(device, &fenceInfo, nullptr, &_frameFence), "vkCreateFence");

    createSwapchain();
    createPresentationShader();

    ImGui_ImplVulkan_InitInfo initInfo = {};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = context.getInstance();
    initInfo.PhysicalDevice = context.getPhysicalDevice();
    initInfo.Device = device;
    initInfo.QueueFamily = context.getQueueFamily();
    initInfo.Queue = context.getQueue();
    initInfo.DescriptorPoolSize = ImGuiDescriptorPoolSize;
    initInfo.MinImageCount = _minImageCount;
    initInfo.ImageCount = std::max(_minImageCount, static_cast<uint32_t>(_swapchainImages.size()));
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &_surfaceFormat.format,
    };
    initInfo.CheckVkResultFn = checkImGuiVkResult;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("The Vulkan backend of the user interface could not be initialized.");
    }
}

void VulkanFrameRenderer::shutdown()
{
    auto& context = VulkanContext::get();
    auto device = context.getDevice();
    context.waitIdle();

    ImGui_ImplVulkan_Shutdown();
    _presentationShader.reset();
    _scene.reset();
    destroySwapchainImageResources();
    vkDestroySwapchainKHR(device, _swapchain, nullptr);
    _swapchain = VK_NULL_HANDLE;

    vkDestroyFence(device, _frameFence, nullptr);
    vkDestroySemaphore(device, _imageAcquiredSemaphore, nullptr);
    vkDestroyCommandPool(device, _commandPool, nullptr);
}

void VulkanFrameRenderer::newFrame()
{
    _clearColor = {0, 0, 0};
    _scene.reset();
    ImGui_ImplVulkan_NewFrame();
}

void VulkanFrameRenderer::clearScreen(FloatColorRGB const& color)
{
    _clearColor = color;
    _scene.reset();
}

void VulkanFrameRenderer::drawScene(SceneUpdateFunc const& updateFunc, SceneRenderFunc const& renderFunc)
{
    _scene = Scene{.updateFunc = updateFunc, .renderFunc = renderFunc};
}

void VulkanFrameRenderer::render(ImDrawData* drawData)
{
    auto& context = VulkanContext::get();
    auto device = context.getDevice();

    // The scene rendering overwrites resources of the previous frame, e.g. the geometry buffers
    if (_frameSubmitted) {
        checkVkResult(vkWaitForFences(device, 1, &_frameFence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        _frameSubmitted = false;
    }
    if (context.getFrameNumber() > 0) {
        context.onFrameCompleted(context.getFrameNumber() - 1);
    }

    // Frames that are not shown still update the scene since the simulation can be synchronized with the rendering
    auto skipFrame = [&] {
        if (_scene) {
            _scene->updateFunc();
        }
        context.advanceFrameNumber();
    };

    auto framebufferSize = getFramebufferSize();
    if (framebufferSize.x <= 0 || framebufferSize.y <= 0) {
        skipFrame();
        return;
    }
    if (_swapchainOutdated || static_cast<uint32_t>(framebufferSize.x) != _extent.width || static_cast<uint32_t>(framebufferSize.y) != _extent.height) {
        context.waitIdle();
        createSwapchain();
    }

    uint32_t imageIndex = 0;
    auto acquireResult = vkAcquireNextImageKHR(device, _swapchain, UINT64_MAX, _imageAcquiredSemaphore, VK_NULL_HANDLE, &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        _swapchainOutdated = true;
        skipFrame();
        return;
    }
    if (acquireResult == VK_SUBOPTIMAL_KHR) {
        _swapchainOutdated = true;
    } else {
        checkVkResult(acquireResult, "vkAcquireNextImageKHR");
    }

    checkVkResult(vkResetCommandBuffer(_commandBuffer, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    checkVkResult(vkBeginCommandBuffer(_commandBuffer, &beginInfo), "vkBeginCommandBuffer");

    VulkanImage* sceneImage = nullptr;
    if (_scene) {
        _scene->updateFunc();
        sceneImage = &_scene->renderFunc(_commandBuffer);
        VulkanContext::useImage(_commandBuffer, *sceneImage, ImageUsage::ShaderRead);
    }

    auto transitionSwapchainImage = [&](VkImageLayout oldLayout,
                                        VkImageLayout newLayout,
                                        VkPipelineStageFlags2 srcStage,
                                        VkAccessFlags2 srcAccess,
                                        VkPipelineStageFlags2 dstStage,
                                        VkAccessFlags2 dstAccess) {
        VkImageMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = srcStage,
            .srcAccessMask = srcAccess,
            .dstStageMask = dstStage,
            .dstAccessMask = dstAccess,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = _swapchainImages.at(imageIndex),
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VkDependencyInfo dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
        vkCmdPipelineBarrier2(_commandBuffer, &dependencyInfo);
    };
    transitionSwapchainImage(
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);

    VkRenderingAttachmentInfo colorAttachment{
        .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = _swapchainImageViews.at(imageIndex),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue = {.color = {.float32 = {_clearColor.r, _clearColor.g, _clearColor.b, 1.0f}}},
    };
    VkRenderingInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, _extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorAttachment,
    };
    vkCmdBeginRendering(_commandBuffer, &renderingInfo);

    if (sceneImage != nullptr) {
        VkViewport viewport{0, 0, static_cast<float>(_extent.width), static_cast<float>(_extent.height), 0.0f, 1.0f};
        VkRect2D scissor{{0, 0}, _extent};
        vkCmdSetViewport(_commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(_commandBuffer, 0, 1, &scissor);
        _presentationShader->setTexture("sceneTexture", *sceneImage);
        _presentationShader->bind(_commandBuffer, PipelineState{.colorFormat = _surfaceFormat.format});
        vkCmdDraw(_commandBuffer, 3, 1, 0, 0);
    }
    if (drawData != nullptr) {
        ImGui_ImplVulkan_RenderDrawData(drawData, _commandBuffer);
    }
    vkCmdEndRendering(_commandBuffer);

    transitionSwapchainImage(
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        VK_ACCESS_2_NONE);
    checkVkResult(vkEndCommandBuffer(_commandBuffer), "vkEndCommandBuffer");

    VkSemaphoreSubmitInfo waitInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = _imageAcquiredSemaphore,
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    VkSemaphoreSubmitInfo signalInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = _renderFinishedSemaphores.at(imageIndex),
        .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
    };
    VkCommandBufferSubmitInfo commandBufferInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = _commandBuffer};
    VkSubmitInfo2 submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount = 1,
        .pWaitSemaphoreInfos = &waitInfo,
        .commandBufferInfoCount = 1,
        .pCommandBufferInfos = &commandBufferInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos = &signalInfo,
    };
    checkVkResult(vkResetFences(device, 1, &_frameFence), "vkResetFences");
    checkVkResult(vkQueueSubmit2(context.getQueue(), 1, &submitInfo, _frameFence), "vkQueueSubmit2");
    _frameSubmitted = true;

    VkPresentInfoKHR presentInfo{
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &_renderFinishedSemaphores.at(imageIndex),
        .swapchainCount = 1,
        .pSwapchains = &_swapchain,
        .pImageIndices = &imageIndex,
    };
    auto presentResult = vkQueuePresentKHR(context.getQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        _swapchainOutdated = true;
    } else {
        checkVkResult(presentResult, "vkQueuePresentKHR");
    }
    context.advanceFrameNumber();
}

VkFormat VulkanFrameRenderer::getScreenFormat() const
{
    return _surfaceFormat.format;
}

void VulkanFrameRenderer::createSwapchain()
{
    auto& context = VulkanContext::get();
    auto physicalDevice = context.getPhysicalDevice();
    auto surface = context.getSurface();
    auto device = context.getDevice();

    VkSurfaceCapabilitiesKHR capabilities;
    checkVkResult(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities), "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    if (_swapchain == VK_NULL_HANDLE) {
        uint32_t numFormats = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &numFormats, nullptr);
        std::vector<VkSurfaceFormatKHR> formats(numFormats);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &numFormats, formats.data());
        if (formats.empty()) {
            throw std::runtime_error("The window surface does not support any format.");
        }

        // Linear formats keep the colors as they were with OpenGL
        _surfaceFormat = formats.front();
        for (auto const& format : formats) {
            if ((format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_R8G8B8A8_UNORM)
                && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                _surfaceFormat = format;
                break;
            }
        }
    }

    auto framebufferSize = getFramebufferSize();
    _extent = capabilities.currentExtent;
    if (_extent.width == UINT32_MAX) {
        _extent.width = std::clamp(static_cast<uint32_t>(framebufferSize.x), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        _extent.height = std::clamp(static_cast<uint32_t>(framebufferSize.y), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    _minImageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        _minImageCount = std::min(_minImageCount, capabilities.maxImageCount);
    }

    auto compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if ((capabilities.supportedCompositeAlpha & compositeAlpha) == 0) {
        for (auto candidate : {VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR, VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR, VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR}) {
            if ((capabilities.supportedCompositeAlpha & candidate) != 0) {
                compositeAlpha = candidate;
                break;
            }
        }
    }

    auto oldSwapchain = _swapchain;
    VkSwapchainCreateInfoKHR swapchainInfo{
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = _minImageCount,
        .imageFormat = _surfaceFormat.format,
        .imageColorSpace = _surfaceFormat.colorSpace,
        .imageExtent = _extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = compositeAlpha,
        .presentMode = VK_PRESENT_MODE_FIFO_KHR,
        .clipped = VK_TRUE,
        .oldSwapchain = oldSwapchain,
    };
    checkVkResult(vkCreateSwapchainKHR(device, &swapchainInfo, nullptr, &_swapchain), "vkCreateSwapchainKHR");
    destroySwapchainImageResources();
    if (oldSwapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, oldSwapchain, nullptr);
    }

    uint32_t numImages = 0;
    vkGetSwapchainImagesKHR(device, _swapchain, &numImages, nullptr);
    _swapchainImages.resize(numImages);
    vkGetSwapchainImagesKHR(device, _swapchain, &numImages, _swapchainImages.data());

    for (auto const& image : _swapchainImages) {
        VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = _surfaceFormat.format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };
        VkImageView view;
        checkVkResult(vkCreateImageView(device, &viewInfo, nullptr, &view), "vkCreateImageView");
        _swapchainImageViews.emplace_back(view);

        VkSemaphoreCreateInfo semaphoreInfo{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkSemaphore semaphore;
        checkVkResult(vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore), "vkCreateSemaphore");
        _renderFinishedSemaphores.emplace_back(semaphore);
    }
    _swapchainOutdated = false;
}

void VulkanFrameRenderer::destroySwapchainImageResources()
{
    auto device = VulkanContext::get().getDevice();
    for (auto const& view : _swapchainImageViews) {
        vkDestroyImageView(device, view, nullptr);
    }
    for (auto const& semaphore : _renderFinishedSemaphores) {
        vkDestroySemaphore(device, semaphore, nullptr);
    }
    _swapchainImages.clear();
    _swapchainImageViews.clear();
    _renderFinishedSemaphores.clear();
}

void VulkanFrameRenderer::createPresentationShader()
{
    _presentationShader = _Shader::createFromSource(PresentationVS, PresentationFS);
}

IntVector2D VulkanFrameRenderer::getFramebufferSize() const
{
    IntVector2D result;
    glfwGetFramebufferSize(_window, &result.x, &result.y);
    return result;
}
