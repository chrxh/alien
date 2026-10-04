#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include <Base/Definitions.h>
#include <Base/Singleton.h>

#include <EngineInterface/Definitions.h>
#include <EngineInterface/GeometryBuffers.h>

#include "Definitions.h"

struct VulkanBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceSize allocationSize = 0;
    bool dedicatedAllocation = false;
    void* mapped = nullptr;
    void* sharedHandle = nullptr;  // Windows
    int sharedFd = -1;             // Linux
};

struct VulkanImage
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    IntVector2D size;
    uint32_t mipLevels = 1;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

enum class VulkanMemory
{
    DeviceLocal,
    DeviceLocalShareable,
    HostVisible,
};

void checkVkResult(VkResult result, char const* operation);

// Owns the Vulkan instance and device and provides the basic resource handling
class VulkanContext
{
    MAKE_SINGLETON(VulkanContext);

public:
    // The device whose UUID matches the one of the GPU engine is preferred, so that both can share memory
    void setup(GLFWwindow* window, std::optional<GpuUuid> const& preferredGpu);
    void shutdown();
    bool isActive() const;

    VkInstance getInstance() const;
    VkPhysicalDevice getPhysicalDevice() const;
    VkDevice getDevice() const;
    uint32_t getQueueFamily() const;
    VkQueue getQueue() const;
    VkSurfaceKHR getSurface() const;
    VkPhysicalDeviceProperties const& getProperties() const;
    VkSampler getLinearClampSampler() const;

    // Black image in shader read layout for samplers without texture
    VulkanImage const& getDummyImage();

    // Whether buffer memory can be exported to the GPU engine
    bool isMemorySharingSupported() const;

    VulkanBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VulkanMemory memory);
    void destroyBuffer(VulkanBuffer& buffer);
    SharedGeometryMemory exportMemory(VulkanBuffer const& buffer);

    VulkanImage createImage(IntVector2D const& size, VkFormat format, VkImageUsageFlags usage, uint32_t mipLevels = 1);

    // Creates an image in shader read layout from RGBA pixels
    VulkanImage createSampledImage(uint8_t const* pixels, IntVector2D const& size);
    void destroyImage(VulkanImage& image);
    static void transitionImage(VkCommandBuffer commandBuffer, VulkanImage& image, VkImageLayout newLayout);
    static void memoryBarrier(VkCommandBuffer commandBuffer);

    // Records the commands into an own command buffer, submits it and waits for its completion
    void submitAndWait(std::function<void(VkCommandBuffer)> const& recordFunc);
    void waitIdle();

    // Waits for the GPU and destroys all resources whose destruction has been postponed
    void releasePendingResources();

    // Descriptor sets that stay valid until the current frame has been rendered
    VkDescriptorSet allocateFrameDescriptorSet(VkDescriptorSetLayout layout);

    // Destroys resources once the GPU has finished the frame being prepared, which may still use them
    void destroyLater(std::function<void()> const& destroyFunc);
    void destroyBufferLater(VulkanBuffer& buffer);
    void destroyImageLater(VulkanImage& image);

    // Called by the frame renderer when all commands of a frame have been executed
    void onFrameCompleted(uint64_t frameNumber);
    uint64_t getFrameNumber() const;
    void advanceFrameNumber();

private:
    void createInstance();
    void selectPhysicalDevice(std::optional<GpuUuid> const& preferredGpu);
    void createDevice();
    void createDescriptorPool();
    void createSampler();

    void exportSharedHandle(VulkanBuffer& buffer);
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const;
    void runPendingDestructions(std::optional<uint64_t> completedFrameNumber);

    bool _active = false;
    bool _validationEnabled = false;
    VkInstance _instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT _debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR _surface = VK_NULL_HANDLE;
    VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties _properties = {};
    VkPhysicalDeviceMemoryProperties _memoryProperties = {};
    bool _memorySharingSupported = false;
    bool _largePointsSupported = false;
    VkDevice _device = VK_NULL_HANDLE;
    uint32_t _queueFamily = 0;
    VkQueue _queue = VK_NULL_HANDLE;
    VkCommandPool _commandPool = VK_NULL_HANDLE;
    VkFence _submitFence = VK_NULL_HANDLE;
    VkSampler _linearClampSampler = VK_NULL_HANDLE;
    std::optional<VulkanImage> _dummyImage;

    std::vector<VkDescriptorPool> _frameDescriptorPools;
    size_t _currentFrameDescriptorPool = 0;

    uint64_t _frameNumber = 0;
    struct PendingDestruction
    {
        uint64_t frameNumber = 0;
        std::function<void()> destroyFunc;
    };
    std::vector<PendingDestruction> _pendingDestructions;
};
