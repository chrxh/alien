
add_library(Rendering
    Definitions.h
    PictureData.h
    RenderGraph.cpp
    RenderGraph.h
    RenderingService.cpp
    RenderingService.h
    RenderStep.cpp
    RenderStep.h
    RenderView.h
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
target_link_libraries(Rendering PUBLIC Base imgui::imgui)
target_link_libraries(Rendering PRIVATE Data EngineInterface Shaders Boost::boost glfw)
target_link_libraries(Rendering PRIVATE Vulkan::Vulkan glslang::glslang glslang::glslang-default-resource-limits glslang::SPIRV)

# GLFW must not pull in the OpenGL headers
target_compile_definitions(Rendering PRIVATE GLFW_INCLUDE_NONE)

target_precompile_headers(Rendering PRIVATE
    <memory>
    <optional>
    <string>
    <vector>
    <imgui.h>
)

if (MSVC)
    target_compile_options(Rendering PRIVATE "/MP")
endif()
