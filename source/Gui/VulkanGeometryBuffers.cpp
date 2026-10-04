#include "VulkanGeometryBuffers.h"

#include <algorithm>
#include <cstring>

#include <Base/GlobalSettings.h>

namespace
{
    auto constexpr BufferUsage =
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
}

VulkanGeometryBuffers _VulkanGeometryBuffers::create()
{
    return VulkanGeometryBuffers(new _VulkanGeometryBuffers());
}

_VulkanGeometryBuffers::~_VulkanGeometryBuffers()
{
    for (auto& buffer : _buffers) {
        VulkanContext::get().destroyBufferLater(buffer);
    }
    for (auto& buffer : _stagingBuffers) {
        VulkanContext::get().destroyBufferLater(buffer);
    }
}

bool _VulkanGeometryBuffers::isMemoryShareable() const
{
    return _shareable;
}

SharedGeometryMemory _VulkanGeometryBuffers::shareMemory(GeometryBufferType type)
{
    return VulkanContext::get().exportMemory(_buffers.at(type));
}

void _VulkanGeometryBuffers::upload(GeometryBufferType type, void const* data, uint64_t sizeInBytes)
{
    auto& stagingBuffer = _stagingBuffers.at(type);
    if (stagingBuffer.buffer == VK_NULL_HANDLE) {
        stagingBuffer = VulkanContext::get().createBuffer(_buffers.at(type).size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanMemory::HostVisible);
    }
    std::memcpy(stagingBuffer.mapped, data, sizeInBytes);
    _pendingUploadSizes.at(type) = std::max(_pendingUploadSizes.at(type), sizeInBytes);
}

void _VulkanGeometryBuffers::download(GeometryBufferType type, void* data, uint64_t sizeInBytes) const
{
    auto const& stagingBuffer = _stagingBuffers.at(type);
    if (stagingBuffer.mapped != nullptr) {
        std::memcpy(data, stagingBuffer.mapped, sizeInBytes);
        return;
    }
    auto& context = VulkanContext::get();
    auto readbackBuffer = context.createBuffer(sizeInBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VulkanMemory::HostVisible);
    context.submitAndWait([&](VkCommandBuffer commandBuffer) {
        VkBufferCopy region{.size = sizeInBytes};
        vkCmdCopyBuffer(commandBuffer, _buffers.at(type).buffer, readbackBuffer.buffer, 1, &region);
    });
    std::memcpy(data, readbackBuffer.mapped, sizeInBytes);
    context.destroyBuffer(readbackBuffer);
}

void _VulkanGeometryBuffers::prepareForRendering(VkCommandBuffer commandBuffer)
{
    auto copied = false;
    for (GeometryBufferType type = 0; type < GeometryBufferType_Count; ++type) {
        auto& pendingUploadSize = _pendingUploadSizes.at(type);
        if (pendingUploadSize > 0) {
            VkBufferCopy region{.size = pendingUploadSize};
            vkCmdCopyBuffer(commandBuffer, _stagingBuffers.at(type).buffer, _buffers.at(type).buffer, 1, &region);
            pendingUploadSize = 0;
            copied = true;
        }
    }

    // The GPU engine has already finished writing the shared memory when the frame is submitted
    if (copied) {
        VkMemoryBarrier2 barrier{
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_ATTRIBUTE_INPUT_BIT | VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT,
        };
        VkDependencyInfo dependencyInfo{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
        vkCmdPipelineBarrier2(commandBuffer, &dependencyInfo);
    }
}

VkBuffer _VulkanGeometryBuffers::getBuffer(GeometryBufferType type) const
{
    return _buffers.at(type).buffer;
}

void _VulkanGeometryBuffers::reallocate(GeometryBufferType type, uint64_t sizeInBytes)
{
    auto& context = VulkanContext::get();
    context.destroyBufferLater(_buffers.at(type));
    context.destroyBufferLater(_stagingBuffers.at(type));
    _pendingUploadSizes.at(type) = 0;

    _buffers.at(type) = context.createBuffer(sizeInBytes, BufferUsage, _shareable ? VulkanMemory::DeviceLocalShareable : VulkanMemory::DeviceLocal);
    if (!_shareable) {
        _stagingBuffers.at(type) = context.createBuffer(sizeInBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VulkanMemory::HostVisible);
    }
}

_VulkanGeometryBuffers::_VulkanGeometryBuffers()
    : _shareable(VulkanContext::get().isMemorySharingSupported() && GlobalSettings::get().isInterop())
{}
