#pragma once

#include <array>
#include <memory>

#include <EngineInterface/GeometryBuffers.h>

#include "VulkanContext.h"

class _VulkanGeometryBuffers;
using VulkanGeometryBuffers = std::shared_ptr<_VulkanGeometryBuffers>;

// Geometry buffers in device local memory. The GPU engine either writes into them directly or the data arrives over a staging buffer.
class _VulkanGeometryBuffers : public _GeometryBuffers
{
public:
    static VulkanGeometryBuffers create();
    ~_VulkanGeometryBuffers() override;

    bool isMemoryShareable() const override;
    SharedGeometryMemory shareMemory(GeometryBufferType type) override;

    void upload(GeometryBufferType type, void const* data, uint64_t sizeInBytes) override;
    void download(GeometryBufferType type, void* data, uint64_t sizeInBytes) const override;

    // Copies the uploaded data to the device local buffers and makes all buffers visible for rendering
    void prepareForRendering(VkCommandBuffer commandBuffer);

    VkBuffer getBuffer(GeometryBufferType type) const;

protected:
    void reallocate(GeometryBufferType type, uint64_t sizeInBytes) override;

private:
    _VulkanGeometryBuffers();

    bool _shareable = false;
    std::array<VulkanBuffer, GeometryBufferType_Count> _buffers;
    std::array<VulkanBuffer, GeometryBufferType_Count> _stagingBuffers;
    std::array<uint64_t, GeometryBufferType_Count> _pendingUploadSizes = {};
};
