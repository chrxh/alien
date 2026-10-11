
add_library(RenderingImpl
    Definitions.h
    RenderGraph.cpp
    RenderGraph.h
    RenderingFacadeImpl.cpp
    RenderingFacadeImpl.h
    RenderStep.cpp
    RenderStep.h
    Shader.cpp
    Shader.h
    SimulationRenderer.cpp
    SimulationRenderer.h
    TextureService.cpp
    TextureService.h
    VulkanContext.cpp
    VulkanContext.h
    VulkanFrameRenderer.cpp
    VulkanFrameRenderer.h
    VulkanGeometryBuffers.cpp
    VulkanGeometryBuffers.h)

# Vulkan stays an implementation detail of this component
target_link_libraries(RenderingImpl PUBLIC RenderingInterface)
target_link_libraries(RenderingImpl PRIVATE Base Data EngineInterface Shaders Boost::boost glfw imgui::imgui)
target_link_libraries(RenderingImpl PRIVATE imgui_impl_vulkan volk::volk glslang::glslang glslang::glslang-default-resource-limits glslang::SPIRV)

# GLFW must not pull in the OpenGL headers
target_compile_definitions(RenderingImpl PRIVATE GLFW_INCLUDE_NONE)

target_precompile_headers(RenderingImpl PRIVATE
    <memory>
    <optional>
    <string>
    <vector>
    <imgui.h>
)

if (MSVC)
    target_compile_options(RenderingImpl PRIVATE "/MP")
endif()
