add_library(RenderingInterface
    Definitions.h
    PictureData.h
    RenderingFacade.cpp
    RenderingFacade.h
    RenderView.h
    TextureData.h)

target_link_libraries(RenderingInterface Base)
target_link_libraries(RenderingInterface imgui::imgui)

if (MSVC)
    target_compile_options(RenderingInterface PRIVATE "/MP")
endif()
