#include "Shader.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <regex>

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <EngineInterface/GeometryBuffers.h>

#include "VulkanContext.h"

namespace
{
    auto const UniformBlockName = std::string("PushConstants");

    struct UniformDeclaration
    {
        std::string type;
        std::string name;
        uint32_t offset = 0;
    };

    // The shaders are written in GLSL as for OpenGL with plain uniforms and implicit interface locations. The translation
    // gathers the plain uniforms of all stages in one push constant block, assigns bindings to the samplers and locations to
    // the stage interfaces. Matching the interfaces by name reproduces the linking of OpenGL.
    struct TranslatedProgram
    {
        std::vector<std::string> stageSources;
        std::vector<UniformDeclaration> uniforms;
        uint32_t uniformBlockSize = 0;
        std::map<std::string, uint32_t> samplerBindings;
        std::set<uint32_t> vertexInputLocations;
    };

    std::pair<uint32_t, uint32_t> getSizeAndAlignment(std::string const& type)
    {
        if (type == "vec2" || type == "ivec2") {
            return {8, 8};
        }
        if (type == "vec3" || type == "ivec3") {
            return {12, 16};
        }
        if (type == "vec4" || type == "ivec4") {
            return {16, 16};
        }
        return {4, 4};
    }

    std::vector<std::string> splitLines(std::string_view source)
    {
        std::vector<std::string> result;
        for (auto const& range : source | std::views::split('\n')) {
            auto line = std::string(range.begin(), range.end());
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            result.emplace_back(std::move(line));
        }
        return result;
    }

    TranslatedProgram translateProgram(std::vector<std::string_view> const& sources)
    {
        static std::regex const uniformRegex(R"(^\s*uniform\s+(\w+)\s+(\w+)\s*;.*$)");
        static std::regex const interfaceRegex(R"(^(\s*)((?:flat|smooth|noperspective)\s+)?(in|out)\s+(\w+)\s+(\w+)\s*(\[\s*\d*\s*\])?\s*;(.*)$)");
        static std::regex const versionRegex(R"(^\s*#version\s+.*$)");
        static std::regex const vertexInputRegex(R"(^\s*layout\s*\(\s*location\s*=\s*(\d+)\s*\)\s*in\s.*$)");

        TranslatedProgram result;
        std::vector<std::vector<std::string>> stageLines;
        for (auto const& source : sources) {
            stageLines.emplace_back(splitLines(source));
        }

        for (auto const& line : stageLines.front()) {
            std::smatch match;
            if (std::regex_match(line, match, vertexInputRegex)) {
                result.vertexInputLocations.insert(static_cast<uint32_t>(std::stoul(match[1].str())));
            }
        }

        for (auto const& lines : stageLines) {
            for (auto const& line : lines) {
                std::smatch match;
                if (!std::regex_match(line, match, uniformRegex)) {
                    continue;
                }
                auto type = match[1].str();
                auto name = match[2].str();
                if (type.starts_with("sampler")) {
                    result.samplerBindings.try_emplace(name, static_cast<uint32_t>(result.samplerBindings.size()));
                } else if (std::ranges::none_of(result.uniforms, [&](auto const& uniform) { return uniform.name == name; })) {
                    auto [size, alignment] = getSizeAndAlignment(type);
                    auto offset = (result.uniformBlockSize + alignment - 1) / alignment * alignment;
                    result.uniforms.emplace_back(UniformDeclaration{.type = type, .name = name, .offset = offset});
                    result.uniformBlockSize = offset + size;
                }
            }
        }

        std::string uniformBlock;
        if (!result.uniforms.empty()) {
            uniformBlock = "layout(push_constant) uniform " + UniformBlockName + " {";
            for (auto const& uniform : result.uniforms) {
                uniformBlock += " layout(offset = " + std::to_string(uniform.offset) + ") " + uniform.type + " " + uniform.name + ";";
            }
            uniformBlock += " };";
        }

        std::map<std::string, int> interfaceLocations;
        for (size_t stage = 0; stage < stageLines.size(); ++stage) {
            auto isFragmentStage = stage == stageLines.size() - 1;
            auto numFragmentOutputs = 0;
            std::string translated;
            for (auto const& line : stageLines.at(stage)) {
                std::smatch match;
                if (std::regex_match(line, versionRegex)) {
                    translated += "#version 450\n" + uniformBlock + "\n";
                } else if (std::regex_match(line, match, uniformRegex)) {
                    auto type = match[1].str();
                    auto name = match[2].str();
                    if (type.starts_with("sampler")) {
                        translated += "layout(set = 0, binding = " + std::to_string(result.samplerBindings.at(name)) + ") uniform " + type + " " + name + ";\n";
                    } else {
                        translated += "\n";
                    }
                } else if (std::regex_match(line, match, interfaceRegex) && !(stage == 0 && match[3].str() == "in")) {
                    auto direction = match[3].str();
                    auto name = match[5].str();
                    int location;
                    if (isFragmentStage && direction == "out") {
                        location = numFragmentOutputs++;
                    } else {
                        location = interfaceLocations.try_emplace(name, static_cast<int>(interfaceLocations.size())).first->second;
                    }
                    translated += match[1].str() + "layout(location = " + std::to_string(location) + ") " + match[2].str() + direction + " " + match[4].str()
                        + " " + name + match[6].str() + ";" + match[7].str() + "\n";
                } else {
                    translated += line + "\n";
                }
            }
            result.stageSources.emplace_back(translated);
        }
        result.uniformBlockSize = (result.uniformBlockSize + 3) / 4 * 4;
        return result;
    }

    void initGlslang()
    {
        [[maybe_unused]] static auto initialized = glslang::InitializeProcess();
    }

    std::vector<uint32_t> compileStage(EShLanguage stage, std::string const& source)
    {
        auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
        auto shader = std::make_unique<glslang::TShader>(stage);
        auto sourcePtr = source.c_str();
        shader->setStrings(&sourcePtr, 1);
        shader->setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 100);
        shader->setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
        // SPIR-V 1.6 would turn discard into a demotion to a helper invocation
        shader->setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_5);
        if (!shader->parse(GetDefaultResources(), 450, false, messages)) {
            throw std::runtime_error(std::string("Shader compilation failed:\n") + shader->getInfoLog() + "\n" + source);
        }
        glslang::TProgram program;
        program.addShader(shader.get());
        if (!program.link(messages)) {
            throw std::runtime_error(std::string("Shader linking failed:\n") + program.getInfoLog());
        }
        std::vector<uint32_t> result;
        glslang::GlslangToSpv(*program.getIntermediate(stage), result);
        return result;
    }

    struct VertexInputDescription
    {
        uint32_t stride = 0;
        std::vector<VkVertexInputAttributeDescription> attributes;
    };

    VertexInputDescription getVertexInputDescription(VertexLayout layout)
    {
        switch (layout) {
        case VertexLayout::FullscreenQuad:
            return {5 * sizeof(float), {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 3 * sizeof(float)}}};
        case VertexLayout::Objects:
            return {
                sizeof(ObjectVertexData),
                {
                    {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ObjectVertexData, pos)},
                    {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ObjectVertexData, color)},
                    {2, 0, VK_FORMAT_R32_SINT, offsetof(ObjectVertexData, state)},
                    {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(ObjectVertexData, highlightIntensity)},
                }};
        case VertexLayout::FluidParticles:
            return {
                sizeof(FluidParticleVertexData),
                {
                    {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(FluidParticleVertexData, pos)},
                    {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(FluidParticleVertexData, color)},
                    {2, 0, VK_FORMAT_R32_SFLOAT, offsetof(FluidParticleVertexData, glow)},
                }};
        case VertexLayout::Locations:
            return {
                sizeof(LocationVertexData),
                {
                    {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LocationVertexData, pos)},
                    {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LocationVertexData, color)},
                    {2, 0, VK_FORMAT_R32_SINT, offsetof(LocationVertexData, shapeType)},
                    {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(LocationVertexData, dimension1)},
                    {4, 0, VK_FORMAT_R32_SFLOAT, offsetof(LocationVertexData, dimension2)},
                    {5, 0, VK_FORMAT_R32_SFLOAT, offsetof(LocationVertexData, fadeoutRadius)},
                    {6, 0, VK_FORMAT_R32_SFLOAT, offsetof(LocationVertexData, opacity)},
                    {7, 0, VK_FORMAT_R32_SINT, offsetof(LocationVertexData, fieldType)},
                    {8, 0, VK_FORMAT_R32_SFLOAT, offsetof(LocationVertexData, fieldParam1)},
                    {9, 0, VK_FORMAT_R32_SFLOAT, offsetof(LocationVertexData, fieldParam2)},
                    {10, 0, VK_FORMAT_R32_SINT, offsetof(LocationVertexData, colored)},
                }};
        case VertexLayout::SelectedObjects:
            return {sizeof(SelectedObjectVertexData), {{0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(SelectedObjectVertexData, pos)}}};
        case VertexLayout::SelectedConnections:
            return {
                sizeof(ConnectionArrowVertexData),
                {
                    {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ConnectionArrowVertexData, pos)},
                    {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ConnectionArrowVertexData, color)},
                    {2, 0, VK_FORMAT_R32_SFLOAT, offsetof(ConnectionArrowVertexData, connectionWeightToObject1)},
                    {3, 0, VK_FORMAT_R32_SFLOAT, offsetof(ConnectionArrowVertexData, connectionWeightToObject2)},
                }};
        case VertexLayout::AttackEvents:
            return {
                sizeof(AttackEventVertexData),
                {
                    {0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(AttackEventVertexData, pos)},
                    {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(AttackEventVertexData, color)},
                }};
        case VertexLayout::DetonationInstances:
            return {
                4 * sizeof(float),
                {
                    {0, 0, VK_FORMAT_R32G32_SFLOAT, 0},
                    {1, 0, VK_FORMAT_R32_SFLOAT, 2 * sizeof(float)},
                    {2, 0, VK_FORMAT_R32_SFLOAT, 3 * sizeof(float)},
                }};
        default:
            return {};
        }
    }

    VkPipelineColorBlendAttachmentState getBlendAttachmentState(BlendMode blendMode)
    {
        VkPipelineColorBlendAttachmentState result{
            .blendEnable = blendMode != BlendMode::None ? VK_TRUE : VK_FALSE,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        auto setFactors = [&result](VkBlendFactor source, VkBlendFactor destination) {
            result.srcColorBlendFactor = source;
            result.dstColorBlendFactor = destination;
            result.srcAlphaBlendFactor = source;
            result.dstAlphaBlendFactor = destination;
        };
        switch (blendMode) {
        case BlendMode::Additive:
            setFactors(VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE);
            break;
        case BlendMode::AlphaAdditive:
            setFactors(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE);
            break;
        case BlendMode::AlphaBlend:
            setFactors(VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA);
            break;
        default:
            setFactors(VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO);
            break;
        }
        return result;
    }
}

Shader _Shader::createFromSource(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource)
{
    return Shader(new _Shader(vertexSource, fragmentSource, geometrySource));
}

_Shader::~_Shader()
{
    auto& context = VulkanContext::get();
    if (!context.isActive()) {
        return;
    }
    context.destroyLater([device = context.getDevice(),
                          pipelines = _pipelines,
                          pipelineLayout = _pipelineLayout,
                          descriptorSetLayout = _descriptorSetLayout,
                          modules = _modules] {
        for (auto const& pipeline : pipelines | std::views::values) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        for (auto const& module : modules | std::views::values) {
            vkDestroyShaderModule(device, module, nullptr);
        }
    });
}

void _Shader::setBool(std::string const& name, bool value)
{
    int intValue = value ? 1 : 0;
    setValue(name, &intValue, sizeof(int));
}

void _Shader::setInt(std::string const& name, int value)
{
    setValue(name, &value, sizeof(int));
}

void _Shader::setFloat(std::string const& name, float value)
{
    setValue(name, &value, sizeof(float));
}

void _Shader::setVec2(std::string const& name, RealVector2D const& value)
{
    float values[] = {value.x, value.y};
    setValue(name, values, sizeof(values));
}

void _Shader::setVec3(std::string const& name, FloatColorRGB const& value)
{
    float values[] = {value.r, value.g, value.b};
    setValue(name, values, sizeof(values));
}

void _Shader::setTexture(std::string const& name, VulkanImage const& image)
{
    if (auto findResult = _samplerBindings.find(name); findResult != _samplerBindings.end()) {
        _boundTextures.insert_or_assign(findResult->second, image.view);
    }
}

bool _Shader::hasTexture(std::string const& name) const
{
    return _samplerBindings.contains(name);
}

void _Shader::bind(VkCommandBuffer commandBuffer, PipelineState const& state)
{
    auto& context = VulkanContext::get();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, getPipeline(state));

    if (!_uniformData.empty()) {
        vkCmdPushConstants(commandBuffer, _pipelineLayout, _stages, 0, static_cast<uint32_t>(_uniformData.size()), _uniformData.data());
    }

    if (!_samplerBindings.empty()) {
        auto descriptorSet = context.allocateFrameDescriptorSet(_descriptorSetLayout);
        std::vector<VkDescriptorImageInfo> imageInfos;
        for (auto const& binding : _samplerBindings | std::views::values) {
            auto findResult = _boundTextures.find(binding);
            auto view = findResult != _boundTextures.end() ? findResult->second : context.getDummyImage().view;
            imageInfos.emplace_back(VkDescriptorImageInfo{context.getLinearClampSampler(), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
        }
        std::vector<VkWriteDescriptorSet> writes;
        for (auto const& [binding, imageInfo] : std::views::zip(_samplerBindings | std::views::values, imageInfos)) {
            writes.emplace_back(VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSet,
                .dstBinding = binding,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &imageInfo,
            });
        }
        vkUpdateDescriptorSets(context.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    }
    _boundTextures.clear();
}

_Shader::_Shader(std::string_view vertexSource, std::string_view fragmentSource, std::string_view geometrySource)
{
    initGlslang();
    auto& context = VulkanContext::get();
    auto device = context.getDevice();

    std::vector<std::string_view> sources{vertexSource};
    std::vector<std::pair<EShLanguage, VkShaderStageFlagBits>> stages{{EShLangVertex, VK_SHADER_STAGE_VERTEX_BIT}};
    if (!geometrySource.empty()) {
        sources.emplace_back(geometrySource);
        stages.emplace_back(EShLangGeometry, VK_SHADER_STAGE_GEOMETRY_BIT);
    }
    sources.emplace_back(fragmentSource);
    stages.emplace_back(EShLangFragment, VK_SHADER_STAGE_FRAGMENT_BIT);

    auto program = translateProgram(sources);
    if (program.uniformBlockSize > context.getProperties().limits.maxPushConstantsSize) {
        throw std::runtime_error("The uniforms of a shader exceed the push constant size of the graphics device.");
    }

    for (auto const& [source, stage] : std::views::zip(program.stageSources, stages)) {
        auto spirv = compileStage(stage.first, source);
        VkShaderModuleCreateInfo moduleInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spirv.size() * sizeof(uint32_t),
            .pCode = spirv.data(),
        };
        VkShaderModule module;
        checkVkResult(vkCreateShaderModule(device, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
        _modules.emplace_back(stage.second, module);
        _stages |= stage.second;
    }

    for (auto const& uniform : program.uniforms) {
        _uniformOffsets.emplace(uniform.name, uniform.offset);
    }
    _uniformData.resize(program.uniformBlockSize, 0);
    _samplerBindings = program.samplerBindings;
    _vertexInputLocations = program.vertexInputLocations;

    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (auto const& binding : _samplerBindings | std::views::values) {
        bindings.emplace_back(VkDescriptorSetLayoutBinding{
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = _stages,
        });
    }
    VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    checkVkResult(vkCreateDescriptorSetLayout(device, &setLayoutInfo, nullptr, &_descriptorSetLayout), "vkCreateDescriptorSetLayout");

    VkPushConstantRange pushConstantRange{.stageFlags = _stages, .offset = 0, .size = program.uniformBlockSize};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &_descriptorSetLayout,
        .pushConstantRangeCount = program.uniformBlockSize > 0 ? 1u : 0u,
        .pPushConstantRanges = &pushConstantRange,
    };
    checkVkResult(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &_pipelineLayout), "vkCreatePipelineLayout");
}

void _Shader::setValue(std::string const& name, void const* data, size_t size)
{
    if (auto findResult = _uniformOffsets.find(name); findResult != _uniformOffsets.end()) {
        std::memcpy(_uniformData.data() + findResult->second, data, size);
    }
}

VkPipeline _Shader::getPipeline(PipelineState const& state)
{
    if (auto findResult = _pipelines.find(state); findResult != _pipelines.end()) {
        return findResult->second;
    }

    std::vector<VkPipelineShaderStageCreateInfo> stageInfos;
    for (auto const& [stage, module] : _modules) {
        stageInfos.emplace_back(VkPipelineShaderStageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = stage,
            .module = module,
            .pName = "main",
        });
    }

    // Attributes the vertex shader does not declare would only cost performance
    auto vertexInput = getVertexInputDescription(state.vertexLayout);
    std::erase_if(vertexInput.attributes, [this](auto const& attribute) { return !_vertexInputLocations.contains(attribute.location); });
    VkVertexInputBindingDescription vertexBinding{.binding = 0, .stride = vertexInput.stride, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = vertexInput.attributes.empty() ? 0u : 1u,
        .pVertexBindingDescriptions = &vertexBinding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInput.attributes.size()),
        .pVertexAttributeDescriptions = vertexInput.attributes.data(),
    };
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = state.topology,
    };
    VkPipelineViewportStateCreateInfo viewportInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    VkPipelineRasterizationStateCreateInfo rasterizationInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    VkPipelineMultisampleStateCreateInfo multisampleInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    VkPipelineDepthStencilStateCreateInfo depthStencilInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = state.depthTest != DepthTest::None ? VK_TRUE : VK_FALSE,
        .depthWriteEnable = state.depthTest != DepthTest::None ? VK_TRUE : VK_FALSE,
        .depthCompareOp = state.depthTest == DepthTest::LessOrEqual ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_LESS,
    };
    auto blendAttachment = getBlendAttachmentState(state.blendMode);
    VkPipelineColorBlendStateCreateInfo colorBlendInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blendAttachment,
    };
    VkDynamicState dynamicStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicStateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2,
        .pDynamicStates = dynamicStates,
    };
    VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &state.colorFormat,
        .depthAttachmentFormat = state.depthFormat,
    };
    VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = static_cast<uint32_t>(stageInfos.size()),
        .pStages = stageInfos.data(),
        .pVertexInputState = &vertexInputInfo,
        .pInputAssemblyState = &inputAssemblyInfo,
        .pViewportState = &viewportInfo,
        .pRasterizationState = &rasterizationInfo,
        .pMultisampleState = &multisampleInfo,
        .pDepthStencilState = &depthStencilInfo,
        .pColorBlendState = &colorBlendInfo,
        .pDynamicState = &dynamicStateInfo,
        .layout = _pipelineLayout,
    };
    VkPipeline result;
    checkVkResult(vkCreateGraphicsPipelines(VulkanContext::get().getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &result), "vkCreateGraphicsPipelines");
    _pipelines.emplace(state, result);
    return result;
}
