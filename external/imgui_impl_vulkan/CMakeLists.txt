# Unchanged Vulkan backend of Dear ImGui 1.91.9 (backends/imgui_impl_vulkan.*). It is built here instead of by the
# vcpkg port so that it calls Vulkan through volk, which loads the Vulkan loader of the system at runtime.
# It has to be replaced together with the ImGui version.
add_library(imgui_impl_vulkan
  imgui_impl_vulkan.cpp)

target_include_directories(imgui_impl_vulkan PUBLIC .)
target_compile_definitions(imgui_impl_vulkan PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)

find_package(imgui CONFIG REQUIRED)
find_package(volk CONFIG REQUIRED)
target_link_libraries(imgui_impl_vulkan PUBLIC imgui::imgui volk::volk)
