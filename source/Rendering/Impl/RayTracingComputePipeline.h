#pragma once

#include <deque>
#include <string>
#include <vector>

#include <volk.h>

#include <Base/Definitions.h>

#include "Definitions.h"

// Compute shader written in GLSL for Vulkan. Pipelines created with the same layout can share their descriptor sets.
class _ComputePipeline
{
public:
    // The work group size has to match the local size declared in the source
    static ComputePipeline create(std::string const& source, VkPipelineLayout layout, IntVector2D const& workGroupSize);
    ~_ComputePipeline();

    // Dispatches at least one invocation per element
    void dispatch(VkCommandBuffer commandBuffer, IntVector2D const& numElements) const;

private:
    _ComputePipeline(std::string const& source, VkPipelineLayout layout, IntVector2D const& workGroupSize);

    VkShaderModule _module = VK_NULL_HANDLE;
    VkPipeline _pipeline = VK_NULL_HANDLE;
    IntVector2D _workGroupSize;
};

// Collects the resources of a descriptor set and writes them in one call
class DescriptorSetWriter
{
public:
    DescriptorSetWriter& uniformBuffer(uint32_t binding, VkBuffer buffer);
    DescriptorSetWriter& storageBuffer(uint32_t binding, VkBuffer buffer);
    DescriptorSetWriter& storageImage(uint32_t binding, VkImageView view);
    DescriptorSetWriter& sampledImage(uint32_t binding, VkImageView view, VkImageLayout layout);
    DescriptorSetWriter& accelerationStructure(uint32_t binding, VkAccelerationStructureKHR accelerationStructure);

    void write(VkDescriptorSet descriptorSet);

private:
    std::vector<VkWriteDescriptorSet> _writes;

    // Deques keep the addresses of their elements when growing
    std::deque<VkDescriptorBufferInfo> _bufferInfos;
    std::deque<VkDescriptorImageInfo> _imageInfos;
    std::deque<VkAccelerationStructureKHR> _accelerationStructures;
    std::deque<VkWriteDescriptorSetAccelerationStructureKHR> _accelerationStructureInfos;
};
