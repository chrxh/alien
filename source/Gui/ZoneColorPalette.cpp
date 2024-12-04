#include "ZoneColorPalette.h"

#include <imgui.h>

ZoneColorPalette::ZoneColorPalette()
{
    for (int n = 0; n < IM_ARRAYSIZE(_palette); n++) {
        ImVec4 color;
        ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.2f, color.x, color.y, color.z);
        color.w = 1.0f;
        _palette[n] = ImColor(color);
    }
}

uint32_t ZoneColorPalette::getColor(int index) const
{
    return _palette[index % IM_ARRAYSIZE(_palette)];
}

auto ZoneColorPalette::getReference()-> Palette&
{
    return _palette;
}
