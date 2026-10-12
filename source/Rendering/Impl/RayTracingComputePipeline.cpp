#include "ComputePipeline.h"

#include "GlslCompiler.h"
#include "VulkanContext.h"

ComputePipeline _ComputePipeline::create(std::string const& source, VkPipelineLayout layout, IntVector2D const& workGroupSize)
{
    return ComputePipeline(new _ComputePipeline(source, layout, workGroupSize));
}

_ComputePipeline::~_ComputePipeline()
{
    auto& context = VulkanContext::get();
    if (!context.isActive()) {
        return;
    }
    context.destroyLater([device = context.getDevice(), pipeline = _pipeline, module = _module] {
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyShaderModule(device, module, nullptr);
    });
}

void _ComputePipeline::dispatch(VkCommandBuffer commandBuffer, IntVector2D const& numElements) const
{
    if (numElements.x <= 0 || numElements.y <= 0) {
        return;
    }
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, _pipeline);
    auto numGroupsX = (numElements.x + _workGroupSize.x - 1) / _workGroupSize.x;
    auto numGroupsY = (numElements.y + _workGroupSize.y - 1) / _workGroupSize.y;
    vkCmdDispatch(commandBuffer, static_cast<uint32_t>(numGroupsX), static_cast<uint32_t>(numGroupsY), 1);
}

_ComputePipeline::_ComputePipeline(std::string const& source, VkPipelineLayout layout, IntVector2D const& workGroupSize)
    : _workGroupSize(workGroupSize)
{
    auto device = VulkanContext::get().getDevice();
    auto spirv = GlslCompiler::compile(VK_SHADER_STAGE_COMPUTE_BIT, source);
    VkShaderModuleCreateInfo moduleInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size() * sizeof(uint32_t),
        .pCode = spirv.data(),
    };
    checkVkResult(vkCreateShaderModule(device, &moduleInfo, nullptr, &_module), "vkCreateShaderModule");

    VkComputePipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage =
            {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = _module,
                .pName = "main",
            },
        .layout = layout,
    };
    auto result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &_pipeline);
    if (result != VK_SUCCESS) {
        vkDestroyShaderModule(device, _module, nullptr);
        checkVkResult(result, "vkCreateComputePipelines");
    }
}

DescriptorSetWriter& DescriptorSetWriter::uniformBuffer(uint32_t binding, VkBuffer buffer)
{
    auto& info = _bufferInfos.emplace_back(VkDescriptorBufferInfo{.buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE});
    _writes.emplace_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &info,
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::storageBuffer(uint32_t binding, VkBuffer buffer)
{
    auto& info = _bufferInfos.emplace_back(VkDescriptorBufferInfo{.buffer = buffer, .offset = 0, .range = VK_WHOLE_SIZE});
    _writes.emplace_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &info,
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::storageImage(uint32_t binding, VkImageView view)
{
    auto& info = _imageInfos.emplace_back(VkDescriptorImageInfo{.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL});
    _writes.emplace_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .pImageInfo = &info,
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::sampledImage(uint32_t binding, VkImageView view, VkImageLayout layout)
{
    auto& info =
        _imageInfos.emplace_back(VkDescriptorImageInfo{.sampler = VulkanContext::get().getLinearClampSampler(), .imageView = view, .imageLayout = layout});
    _writes.emplace_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &info,
    });
    return *this;
}

DescriptorSetWriter& DescriptorSetWriter::accelerationStructure(uint32_t binding, VkAccelerationStructureKHR accelerationStructure)
{
    auto& handle = _accelerationStructures.emplace_back(accelerationStructure);
    auto& info = _accelerationStructureInfos.emplace_back(VkWriteDescriptorSetAccelerationStructureKHR{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures = &handle,
    });
    _writes.emplace_back(VkWriteDescriptorSet{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = &info,
        .dstBinding = binding,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
    });
    return *this;
}

void DescriptorSetWriter::write(VkDescriptorSet descriptorSet)
{
    for (auto& write : _writes) {
        write.dstSet = descriptorSet;
    }
    vkUpdateDescriptorSets(VulkanContext::get().getDevice(), static_cast<uint32_t>(_writes.size()), _writes.data(), 0, nullptr);
}
