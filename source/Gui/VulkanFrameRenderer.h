#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include "Definitions.h"
#include "VulkanContext.h"

struct ImDrawData;

// Presents the frames on the window: the scene of the simulation as background and the user interface on top
class VulkanFrameRenderer
{
    MAKE_SINGLETON(VulkanFrameRenderer);

public:
    void setup(GLFWwindow* window);
    void shutdown();

    void newFrame();

    // Fills the screen of the current frame with a color
    void clearScreen(FloatColorRGB const& color);

    // The update function writes the data of the scene into resources that the GPU no longer uses at this point. It is also called for frames
    // that are not shown, e.g. while the window is minimized. The render function records the scene into its own image, which then fills
    // the screen of the current frame. Its rows are ordered from bottom to top as in OpenGL.
    using SceneUpdateFunc = std::function<void()>;
    using SceneRenderFunc = std::function<VulkanImage&(VkCommandBuffer)>;
    void drawScene(SceneUpdateFunc const& updateFunc, SceneRenderFunc const& renderFunc);

    void render(ImDrawData* drawData);

    VkFormat getScreenFormat() const;

private:
    void createSwapchain();
    void destroySwapchain();
    void createPresentationShader();

    IntVector2D getFramebufferSize() const;

    GLFWwindow* _window = nullptr;
    VkSwapchainKHR _swapchain = VK_NULL_HANDLE;
    VkSurfaceFormatKHR _surfaceFormat = {};
    VkExtent2D _extent = {};
    uint32_t _minImageCount = 2;
    std::vector<VkImage> _swapchainImages;
    std::vector<VkImageView> _swapchainImageViews;
    std::vector<VkImageLayout> _swapchainImageLayouts;
    std::vector<VkSemaphore> _renderFinishedSemaphores;
    bool _swapchainOutdated = false;

    VkCommandPool _commandPool = VK_NULL_HANDLE;
    VkCommandBuffer _commandBuffer = VK_NULL_HANDLE;
    VkSemaphore _imageAcquiredSemaphore = VK_NULL_HANDLE;
    VkFence _frameFence = VK_NULL_HANDLE;
    bool _frameSubmitted = false;

    Shader _presentationShader;

    FloatColorRGB _clearColor = {0, 0, 0};

    struct Scene
    {
        SceneUpdateFunc updateFunc;
        SceneRenderFunc renderFunc;
    };
    std::optional<Scene> _scene;
};
