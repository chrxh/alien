#include "GlslCompiler.h"

#include <memory>
#include <stdexcept>

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <Base/Definitions.h>

namespace
{
    void initGlslang()
    {
        [[maybe_unused]] static auto initialized = glslang::InitializeProcess();
    }

    EShLanguage getLanguage(VkShaderStageFlagBits stage)
    {
        switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT:
            return EShLangVertex;
        case VK_SHADER_STAGE_GEOMETRY_BIT:
            return EShLangGeometry;
        case VK_SHADER_STAGE_FRAGMENT_BIT:
            return EShLangFragment;
        case VK_SHADER_STAGE_COMPUTE_BIT:
            return EShLangCompute;
        default:
            THROW_NOT_IMPLEMENTED();
        }
    }
}

std::vector<uint32_t> GlslCompiler::compile(VkShaderStageFlagBits stage, std::string const& source)
{
    initGlslang();
    auto language = getLanguage(stage);
    auto messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    auto shader = std::make_unique<glslang::TShader>(language);
    auto sourcePtr = source.c_str();
    shader->setStrings(&sourcePtr, 1);
    shader->setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClientVulkan, 100);
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
    glslang::GlslangToSpv(*program.getIntermediate(language), result);
    return result;
}
