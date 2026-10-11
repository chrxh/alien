#pragma once

#include <map>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <volk.h>

#include <Base/Definitions.h>

#include "Definitions.h"

struct VulkanImage;

enum class VertexLayout
{
    None,
    FullscreenQuad,
    Objects,
    FluidParticles,
    Locations,
    SelectedObjects,
    SelectedConnections,
    AttackEvents,
    DetonationInstances,
};

enum class BlendMode
{
    None,
    Additive,
    AlphaAdditive,
    AlphaBlend,
};

enum class DepthTest
{
    None,
    Less,
    LessOrEqual,
};

struct PipelineState
{
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VertexLayout vertexLayout = VertexLayout::None;
    BlendMode blendMode = BlendMode::None;
    DepthTest depthTest = DepthTest::None;
    VkFormat colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat depthFormat = VK_FORMAT_UNDEFINED;

    auto operator<=>(PipelineState const&) const = default;
};

// GLSL program compiled for Vulkan. Plain uniforms become push constants, samplers are bound by name.
class _Shader
{
public:
    static Shader createFromSource(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource = "");
    ~_Shader();

    void setBool(std::string const& name, bool value);
    void setInt(std::string const& name, int value);
    void setFloat(std::string const& name, float value);
    void setVec2(std::string const& name, RealVector2D const& value);
    void setVec3(std::string const& name, FloatColorRGB const& value);

    // The image must be in shader read layout when the draw call is executed
    void setTexture(std::string const& name, VulkanImage const& image);

    // Binds the pipeline, the uniforms and the textures for the next draw call
    void bind(VkCommandBuffer commandBuffer, PipelineState const& state);

private:
    _Shader(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource);

    void setValue(std::string const& name, void const* data, size_t size);
    VkPipeline getPipeline(PipelineState const& state);

    std::vector<std::pair<VkShaderStageFlagBits, VkShaderModule>> _modules;
    VkDescriptorSetLayout _descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout _pipelineLayout = VK_NULL_HANDLE;
    std::map<PipelineState, VkPipeline> _pipelines;

    std::unordered_map<std::string, uint32_t> _uniformOffsets;
    std::vector<uint8_t> _uniformData;
    VkShaderStageFlags _stages = 0;

    std::map<std::string, uint32_t> _samplerBindings;
    std::map<uint32_t, VkImageView> _boundTextures;
    std::set<uint32_t> _vertexInputLocations;
};
