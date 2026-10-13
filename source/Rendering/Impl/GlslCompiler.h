#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <volk.h>

class GlslCompiler
{
public:
    // Translates GLSL written for Vulkan into SPIR-V. Throws a std::runtime_error containing the compiler log if the source is invalid.
    static std::vector<uint32_t> compile(VkShaderStageFlagBits stage, std::string const& source);
};
