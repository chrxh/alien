#include "VulkanContext.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <cstring>
#include <ranges>
#include <stdexcept>

#include <vulkan/vulkan.h>
#if defined(_WIN32)
#include <vulkan/vulkan_win32.h>
#endif

#include <GLFW/glfw3.h>

#include <Base/GlobalSettings.h>
#include <Base/LoggingService.h>
#include <Base/Resources.h>

namespace
{
    auto constexpr ApiVersion = VK_API_VERSION_1_3;
    auto constexpr ValidationLayerName = "VK_LAYER_KHRONOS_validation";
    auto constexpr FrameDescriptorPoolSize = 512;
    auto constexpr ImageWriteAccesses =
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

#if defined(_WIN32)
    auto constexpr ExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    auto constexpr ExternalMemoryExtensionName = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
#else
    auto constexpr ExternalMemoryHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto constexpr ExternalMemoryExtensionName = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
#endif
}

void checkVkResult(VkResult result, char const* operation)
{
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string("Vulkan error ") + std::to_string(result) + " in " + operation + ".");
    }
}

void VulkanContext::setup(GLFWwindow* window, std::optional<GpuUuid> const& preferredGpu)
{
    if (!glfwVulkanSupported()) {
        throw std::runtime_error("Vulkan is not supported on this system. Please update your graphics driver.");
    }
    createInstance();
    checkVkResult(glfwCreateWindowSurface(_instance, window, nullptr, &_surface), "glfwCreateWindowSurface");
    selectPhysicalDevice(preferredGpu);
    createDevice();
    createSampler();
    _active = true;
}

void VulkanContext::shutdown()
{
    if (!_active) {
        return;
    }
    waitIdle();
    runPendingDestructions(std::nullopt);

    for (auto const& pool : _frameDescriptorPools) {
        vkDestroyDescriptorPool(_device, pool, nullptr);
    }
    _frameDescriptorPools.clear();
    if (_dummyImage) {
        destroyImage(*_dummyImage);
        _dummyImage.reset();
    }
    vkDestroySampler(_device, _linearClampSampler, nullptr);
    vkDestroyFence(_device, _submitFence, nullptr);
    vkDestroyCommandPool(_device, _commandPool, nullptr);
    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
    if (_debugMessenger != VK_NULL_HANDLE) {
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(_instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroyMessenger != nullptr) {
            destroyMessenger(_instance, _debugMessenger, nullptr);
        }
    }
    vkDestroyInstance(_instance, nullptr);
    _active = false;
}

bool VulkanContext::isActive() const
{
    return _active;
}

VkInstance VulkanContext::getInstance() const
{
    return _instance;
}

VkPhysicalDevice VulkanContext::getPhysicalDevice() const
{
    return _physicalDevice;
}

VkDevice VulkanContext::getDevice() const
{
    return _device;
}

uint32_t VulkanContext::getQueueFamily() const
{
    return _queueFamily;
}

VkQueue VulkanContext::getQueue() const
{
    return _queue;
}

VkSurfaceKHR VulkanContext::getSurface() const
{
    return _surface;
}

VkPhysicalDeviceProperties const& VulkanContext::getProperties() const
{
    return _properties;
}

VkSampler VulkanContext::getLinearClampSampler() const
{
    return _linearClampSampler;
}

VulkanImage const& VulkanContext::getDummyImage()
{
    if (!_dummyImage) {
        auto image = createImage({1, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        submitAndWait([&image](VkCommandBuffer commandBuffer) {
            useImage(commandBuffer, image, ImageUsage::TransferDestination);
            VkClearColorValue black{.float32 = {0.0f, 0.0f, 0.0f, 1.0f}};
            VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(commandBuffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
            useImage(commandBuffer, image, ImageUsage::ShaderRead);
        });
        _dummyImage = image;
    }
    return *_dummyImage;
}

bool VulkanContext::isMemorySharingSupported() const
{
    return _memorySharingSupported;
}

VulkanBuffer VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VulkanMemory memory)
{
    VulkanBuffer result;
    result.size = size;

    auto shareable = memory == VulkanMemory::DeviceLocalShareable;
    VkExternalMemoryBufferCreateInfo externalBufferInfo{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = static_cast<VkExternalMemoryHandleTypeFlags>(ExternalMemoryHandleType),
    };
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = shareable ? &externalBufferInfo : nullptr,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    checkVkResult(vkCreateBuffer(_device, &bufferInfo, nullptr, &result.buffer), "vkCreateBuffer");

    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(_device, result.buffer, &requirements);

    auto properties =
        memory == VulkanMemory::HostVisible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkExportMemoryAllocateInfo exportInfo{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = static_cast<VkExternalMemoryHandleTypeFlags>(ExternalMemoryHandleType),
    };
    VkMemoryDedicatedAllocateInfo dedicatedInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &exportInfo,
        .buffer = result.buffer,
    };
    if (shareable) {
        VkPhysicalDeviceExternalBufferInfo externalInfo{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            .usage = usage,
            .handleType = ExternalMemoryHandleType,
        };
        VkExternalBufferProperties externalProperties{.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
        vkGetPhysicalDeviceExternalBufferProperties(_physicalDevice, &externalInfo, &externalProperties);
        result.dedicatedAllocation = (externalProperties.externalMemoryProperties.externalMemoryFeatures & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0;
    }

    VkMemoryAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = shareable ? (result.dedicatedAllocation ? static_cast<void const*>(&dedicatedInfo) : static_cast<void const*>(&exportInfo)) : nullptr,
        .allocationSize = requirements.size,
        .memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties),
    };
    checkVkResult(vkAllocateMemory(_device, &allocateInfo, nullptr, &result.memory), "vkAllocateMemory");
    result.allocationSize = requirements.size;
    checkVkResult(vkBindBufferMemory(_device, result.buffer, result.memory, 0), "vkBindBufferMemory");

    if (memory == VulkanMemory::HostVisible) {
        checkVkResult(vkMapMemory(_device, result.memory, 0, VK_WHOLE_SIZE, 0, &result.mapped), "vkMapMemory");
    }
    if (shareable) {
        exportSharedHandle(result);
    }
    return result;
}

void VulkanContext::destroyBuffer(VulkanBuffer& buffer)
{
    if (_active) {
        if (buffer.mapped != nullptr) {
            vkUnmapMemory(_device, buffer.memory);
        }
        vkDestroyBuffer(_device, buffer.buffer, nullptr);
        vkFreeMemory(_device, buffer.memory, nullptr);
    }
#if defined(_WIN32)
    if (buffer.sharedHandle != nullptr) {
        CloseHandle(buffer.sharedHandle);
    }
#else
    if (buffer.sharedFd != -1) {
        close(buffer.sharedFd);
    }
#endif
    buffer = VulkanBuffer();
}

SharedGeometryMemory VulkanContext::exportMemory(VulkanBuffer const& buffer)
{
    SharedGeometryMemory result{.allocationSize = buffer.allocationSize, .dedicatedAllocation = buffer.dedicatedAllocation};
#if defined(_WIN32)
    result.win32Handle = buffer.sharedHandle;
#else
    result.fd = dup(buffer.sharedFd);
#endif
    return result;
}

VulkanImage VulkanContext::createImage(IntVector2D const& size, VkFormat format, VkImageUsageFlags usage, uint32_t mipLevels)
{
    VulkanImage result;
    result.format = format;
    result.size = size;
    result.mipLevels = mipLevels;
    auto isDepth = format == VK_FORMAT_D32_SFLOAT;
    result.aspect = isDepth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageCreateInfo imageInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent = {static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y), 1},
        .mipLevels = mipLevels,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    checkVkResult(vkCreateImage(_device, &imageInfo, nullptr, &result.image), "vkCreateImage");

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(_device, result.image, &requirements);
    VkMemoryAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = requirements.size,
        .memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
    };
    checkVkResult(vkAllocateMemory(_device, &allocateInfo, nullptr, &result.memory), "vkAllocateMemory");
    checkVkResult(vkBindImageMemory(_device, result.image, result.memory, 0), "vkBindImageMemory");

    VkImageViewCreateInfo viewInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = result.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .subresourceRange = {result.aspect, 0, mipLevels, 0, 1},
    };
    checkVkResult(vkCreateImageView(_device, &viewInfo, nullptr, &result.view), "vkCreateImageView");
    return result;
}

VulkanImage VulkanContext::createSampledImage(uint8_t const* pixels, IntVector2D const& size)
{
    auto sizeInBytes = static_cast<VkDeviceSize>(size.x) * size.y * 4;
    auto stagingBuffer = createBuffer(sizeInBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanMemory::HostVisible);
    std::memcpy(stagingBuffer.mapped, pixels, sizeInBytes);

    auto result = createImage(size, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    submitAndWait([&](VkCommandBuffer commandBuffer) {
        useImage(commandBuffer, result, ImageUsage::TransferDestination);
        VkBufferImageCopy region{
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y), 1},
        };
        vkCmdCopyBufferToImage(commandBuffer, stagingBuffer.buffer, result.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        useImage(commandBuffer, result, ImageUsage::ShaderRead);
    });
    destroyBuffer(stagingBuffer);
    return result;
}

void VulkanContext::destroyImage(VulkanImage& image)
{
    if (_active) {
        vkDestroyImageView(_device, image.view, nullptr);
        vkDestroyImage(_device, image.image, nullptr);
        vkFreeMemory(_device, image.memory, nullptr);
    }
    image = VulkanImage();
}

namespace
{
    struct ImageAccess
    {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_NONE;
        VkAccessFlags2 accesses = VK_ACCESS_2_NONE;
    };

    ImageAccess getImageAccess(ImageUsage usage)
    {
        switch (usage) {
        case ImageUsage::ColorAttachment:
            return {
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
        case ImageUsage::DepthAttachment:
            return {
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
        case ImageUsage::ShaderRead:
            return {VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
        case ImageUsage::TransferSource:
            return {VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT};
        case ImageUsage::TransferDestination:
            return {VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
        }
        THROW_NOT_IMPLEMENTED();
    }
}

VulkanImageBarriers& VulkanImageBarriers::add(VulkanImage& image, ImageUsage usage)
{
    auto access = getImageAccess(usage);
    auto pendingWrites = image.lastAccesses & ImageWriteAccesses;
    if (image.layout == access.layout && pendingWrites == 0 && (access.accesses & ImageWriteAccesses) == 0) {
        image.lastStages |= access.stages;
        image.lastAccesses |= access.accesses;
        return *this;
    }
    _barriers.emplace_back(VkImageMemoryBarrier2{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = image.lastStages,
        .srcAccessMask = pendingWrites,
        .dstStageMask = access.stages,
        .dstAccessMask = access.accesses,
        .oldLayout = image.layout,
        .newLayout = access.layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image.image,
        .subresourceRange = {image.aspect, 0, image.mipLevels, 0, 1},
    });
    image.layout = access.layout;
    image.lastStages = access.stages;
    image.lastAccesses = access.accesses;
    return *this;
}

void VulkanImageBarriers::record(VkCommandBuffer commandBuffer)
{
    if (_barriers.empty()) {
        return;
    }
    VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = static_cast<uint32_t>(_barriers.size()),
        .pImageMemoryBarriers = _barriers.data(),
    };
    vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    _barriers.clear();
}

void VulkanContext::useImage(VkCommandBuffer commandBuffer, VulkanImage& image, ImageUsage usage)
{
    VulkanImageBarriers().add(image, usage).record(commandBuffer);
}

void VulkanContext::submitAndWait(std::function<void(VkCommandBuffer)> const& recordFunc)
{
    VkCommandBufferAllocateInfo allocateInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = _commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer commandBuffer;
    checkVkResult(vkAllocateCommandBuffers(_device, &allocateInfo, &commandBuffer), "vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    checkVkResult(vkBeginCommandBuffer(commandBuffer, &beginInfo), "vkBeginCommandBuffer");
    try {
        recordFunc(commandBuffer);
    } catch (...) {
        vkEndCommandBuffer(commandBuffer);
        vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
        throw;
    }
    checkVkResult(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");

    VkSubmitInfo submitInfo{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &commandBuffer,
    };
    checkVkResult(vkResetFences(_device, 1, &_submitFence), "vkResetFences");
    checkVkResult(vkQueueSubmit(_queue, 1, &submitInfo, _submitFence), "vkQueueSubmit");
    checkVkResult(vkWaitForFences(_device, 1, &_submitFence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
}

void VulkanContext::waitIdle()
{
    if (_active) {
        vkDeviceWaitIdle(_device);
    }
}

void VulkanContext::releasePendingResources()
{
    waitIdle();
    runPendingDestructions(std::nullopt);
}

VkDescriptorSet VulkanContext::allocateFrameDescriptorSet(VkDescriptorSetLayout layout)
{
    while (true) {
        if (_currentFrameDescriptorPool == _frameDescriptorPools.size()) {
            createDescriptorPool();
        }
        VkDescriptorSetAllocateInfo allocateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = _frameDescriptorPools.at(_currentFrameDescriptorPool),
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        VkDescriptorSet result;
        auto allocateResult = vkAllocateDescriptorSets(_device, &allocateInfo, &result);
        if (allocateResult == VK_SUCCESS) {
            return result;
        }
        if (allocateResult != VK_ERROR_OUT_OF_POOL_MEMORY && allocateResult != VK_ERROR_FRAGMENTED_POOL) {
            checkVkResult(allocateResult, "vkAllocateDescriptorSets");
        }
        ++_currentFrameDescriptorPool;
    }
}

void VulkanContext::destroyLater(std::function<void()> const& destroyFunc)
{
    if (!_active) {
        return;
    }
    _pendingDestructions.emplace_back(_frameNumber, destroyFunc);
}

void VulkanContext::destroyBufferLater(VulkanBuffer& buffer)
{
    if (buffer.buffer != VK_NULL_HANDLE) {
        destroyLater([this, buffer]() mutable { destroyBuffer(buffer); });
    }
    buffer = VulkanBuffer();
}

void VulkanContext::destroyImageLater(VulkanImage& image)
{
    if (image.image != VK_NULL_HANDLE) {
        destroyLater([this, image]() mutable { destroyImage(image); });
    }
    image = VulkanImage();
}

void VulkanContext::onFrameCompleted(uint64_t frameNumber)
{
    runPendingDestructions(frameNumber);

    for (auto const& pool : _frameDescriptorPools) {
        vkResetDescriptorPool(_device, pool, 0);
    }
    _currentFrameDescriptorPool = 0;
}

uint64_t VulkanContext::getFrameNumber() const
{
    return _frameNumber;
}

void VulkanContext::advanceFrameNumber()
{
    ++_frameNumber;
}

namespace
{
    bool isLayerAvailable(char const* layerName)
    {
        uint32_t numLayers = 0;
        vkEnumerateInstanceLayerProperties(&numLayers, nullptr);
        std::vector<VkLayerProperties> layers(numLayers);
        vkEnumerateInstanceLayerProperties(&numLayers, layers.data());
        return std::ranges::any_of(layers, [&](auto const& layer) { return std::strcmp(layer.layerName, layerName) == 0; });
    }

    VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT,
        VkDebugUtilsMessengerCallbackDataEXT const* callbackData,
        void*)
    {
        auto priority = severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT ? Priority::Important : Priority::Unimportant;
        log(priority, std::string("Vulkan: ") + callbackData->pMessage);
        return VK_FALSE;
    }
}

void VulkanContext::createInstance()
{
    uint32_t numGlfwExtensions = 0;
    auto glfwExtensions = glfwGetRequiredInstanceExtensions(&numGlfwExtensions);
    if (glfwExtensions == nullptr) {
        throw std::runtime_error("The Vulkan surface extensions are not available.");
    }
    std::vector<char const*> extensions(glfwExtensions, glfwExtensions + numGlfwExtensions);

    std::vector<char const*> layers;
    _validationEnabled = GlobalSettings::get().isDebugMode() && isLayerAvailable(ValidationLayerName);
    if (_validationEnabled) {
        layers.emplace_back(ValidationLayerName);
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        log(Priority::Important, "Vulkan validation layer enabled");
    }

    VkApplicationInfo applicationInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "alien",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "alien",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = ApiVersion,
    };
    VkInstanceCreateInfo instanceInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = static_cast<uint32_t>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    auto result = vkCreateInstance(&instanceInfo, nullptr, &_instance);
    if (result == VK_ERROR_INCOMPATIBLE_DRIVER) {
        throw std::runtime_error("Vulkan 1.3 is not supported by the graphics driver. Please update your graphics driver.");
    }
    checkVkResult(result, "vkCreateInstance");

    if (_validationEnabled) {
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(_instance, "vkCreateDebugUtilsMessengerEXT"));
        VkDebugUtilsMessengerCreateInfoEXT messengerInfo{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType =
                VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debugCallback,
        };
        if (createMessenger != nullptr) {
            createMessenger(_instance, &messengerInfo, nullptr, &_debugMessenger);
        }
    }
}

namespace
{
    bool hasExtension(std::vector<VkExtensionProperties> const& extensions, char const* name)
    {
        return std::ranges::any_of(extensions, [&](auto const& extension) { return std::strcmp(extension.extensionName, name) == 0; });
    }

    std::vector<VkExtensionProperties> getDeviceExtensions(VkPhysicalDevice device)
    {
        uint32_t numExtensions = 0;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &numExtensions, nullptr);
        std::vector<VkExtensionProperties> result(numExtensions);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &numExtensions, result.data());
        return result;
    }

    std::optional<uint32_t> findGraphicsAndPresentQueueFamily(VkPhysicalDevice device, VkSurfaceKHR surface)
    {
        uint32_t numFamilies = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &numFamilies, nullptr);
        std::vector<VkQueueFamilyProperties> families(numFamilies);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &numFamilies, families.data());
        for (uint32_t index = 0; index < numFamilies; ++index) {
            VkBool32 presentSupported = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface, &presentSupported);
            if ((families.at(index).queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && presentSupported == VK_TRUE) {
                return index;
            }
        }
        return std::nullopt;
    }

    GpuUuid getDeviceUuid(VkPhysicalDevice device)
    {
        VkPhysicalDeviceIDProperties idProperties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &idProperties};
        vkGetPhysicalDeviceProperties2(device, &properties);
        GpuUuid result;
        std::memcpy(result.data(), idProperties.deviceUUID, result.size());
        return result;
    }
}

void VulkanContext::selectPhysicalDevice(std::optional<GpuUuid> const& preferredGpu)
{
    uint32_t numDevices = 0;
    vkEnumeratePhysicalDevices(_instance, &numDevices, nullptr);
    std::vector<VkPhysicalDevice> devices(numDevices);
    vkEnumeratePhysicalDevices(_instance, &numDevices, devices.data());

    auto bestScore = -1;
    for (auto const& device : devices) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(device, &properties);
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceFeatures(device, &features);
        auto extensions = getDeviceExtensions(device);
        auto queueFamily = findGraphicsAndPresentQueueFamily(device, _surface);

        if (properties.apiVersion < ApiVersion || !queueFamily || features.geometryShader != VK_TRUE
            || !hasExtension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
            log(Priority::Important, std::string("Vulkan device ") + properties.deviceName + " is not suitable");
            continue;
        }
        auto score = 0;
        if (preferredGpu && getDeviceUuid(device) == *preferredGpu) {
            score += 1000;
        }
        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 100;
        }
        if (score > bestScore) {
            bestScore = score;
            _physicalDevice = device;
            _queueFamily = *queueFamily;
        }
    }
    if (_physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("No graphics device with Vulkan 1.3 and geometry shader support found. Please update your graphics driver.");
    }

    vkGetPhysicalDeviceProperties(_physicalDevice, &_properties);
    vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &_memoryProperties);
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(_physicalDevice, &features);
    _largePointsSupported = features.largePoints == VK_TRUE;

    auto extensions = getDeviceExtensions(_physicalDevice);
    auto sameGpuAsEngine = preferredGpu.has_value() && getDeviceUuid(_physicalDevice) == *preferredGpu;
    _memorySharingSupported = sameGpuAsEngine && hasExtension(extensions, ExternalMemoryExtensionName);

    log(Priority::Important, std::string("Vulkan device ") + _properties.deviceName + " selected");
    if (!sameGpuAsEngine) {
        log(Priority::Important, "the Vulkan device differs from the CUDA device, rendering data takes the detour over host memory");
    }
}

void VulkanContext::createDevice()
{
    auto availableExtensions = getDeviceExtensions(_physicalDevice);
    std::vector<char const*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (_memorySharingSupported) {
        extensions.emplace_back(ExternalMemoryExtensionName);
    }
    if (hasExtension(availableExtensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
        extensions.emplace_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = _queueFamily,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };
    VkPhysicalDeviceVulkan13Features features13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,
    };
    VkPhysicalDeviceFeatures2 features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &features13,
    };
    features.features.geometryShader = VK_TRUE;
    features.features.largePoints = _largePointsSupported ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };
    checkVkResult(vkCreateDevice(_physicalDevice, &deviceInfo, nullptr, &_device), "vkCreateDevice");
    vkGetDeviceQueue(_device, _queueFamily, 0, &_queue);

    VkCommandPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = _queueFamily,
    };
    checkVkResult(vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool), "vkCreateCommandPool");

    VkFenceCreateInfo fenceInfo{.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    checkVkResult(vkCreateFence(_device, &fenceInfo, nullptr, &_submitFence), "vkCreateFence");
}

void VulkanContext::createDescriptorPool()
{
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, FrameDescriptorPoolSize * 4};
    VkDescriptorPoolCreateInfo poolInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = FrameDescriptorPoolSize,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    VkDescriptorPool pool;
    checkVkResult(vkCreateDescriptorPool(_device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    _frameDescriptorPools.emplace_back(pool);
}

void VulkanContext::createSampler()
{
    VkSamplerCreateInfo samplerInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxLod = 0.0f,
    };
    checkVkResult(vkCreateSampler(_device, &samplerInfo, nullptr, &_linearClampSampler), "vkCreateSampler");
}

void VulkanContext::exportSharedHandle(VulkanBuffer& buffer)
{
    // The NVIDIA driver rejects every further import of the memory into CUDA once its first exported handle is closed.
    // Therefore, the handle stays open as long as the buffer exists.
#if defined(_WIN32)
    auto getHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(_device, "vkGetMemoryWin32HandleKHR"));
    if (getHandle == nullptr) {
        throw std::runtime_error("vkGetMemoryWin32HandleKHR is not available.");
    }
    VkMemoryGetWin32HandleInfoKHR handleInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
        .memory = buffer.memory,
        .handleType = ExternalMemoryHandleType,
    };
    HANDLE handle = nullptr;
    checkVkResult(getHandle(_device, &handleInfo, &handle), "vkGetMemoryWin32HandleKHR");
    buffer.sharedHandle = handle;
#else
    auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(_device, "vkGetMemoryFdKHR"));
    if (getFd == nullptr) {
        throw std::runtime_error("vkGetMemoryFdKHR is not available.");
    }
    VkMemoryGetFdInfoKHR fdInfo{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = buffer.memory,
        .handleType = ExternalMemoryHandleType,
    };
    checkVkResult(getFd(_device, &fdInfo, &buffer.sharedFd), "vkGetMemoryFdKHR");
#endif
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags properties) const
{
    // Host visible memory should not come from the small device local heap that is mapped into the address space
    auto undesiredProperties = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : 0u;
    std::optional<uint32_t> fallback;
    for (uint32_t index = 0; index < _memoryProperties.memoryTypeCount; ++index) {
        auto flags = _memoryProperties.memoryTypes[index].propertyFlags;
        if ((typeBits & (1u << index)) == 0 || (flags & properties) != properties) {
            continue;
        }
        if ((flags & undesiredProperties) == 0) {
            return index;
        }
        if (!fallback) {
            fallback = index;
        }
    }
    if (fallback) {
        return *fallback;
    }
    throw std::runtime_error("No suitable Vulkan memory type found.");
}

void VulkanContext::runPendingDestructions(std::optional<uint64_t> completedFrameNumber)
{
    auto pendingDestructions = std::move(_pendingDestructions);
    _pendingDestructions.clear();
    for (auto& pending : pendingDestructions) {
        if (!completedFrameNumber || pending.frameNumber <= *completedFrameNumber) {
            pending.destroyFunc();
        } else {
            _pendingDestructions.emplace_back(std::move(pending));
        }
    }
}
