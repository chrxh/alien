#pragma once

#include "Definitions.h"

#include <cstdint>

#include "imgui.h"

namespace Const
{
    int64_t const SimulationSliderColor_Base = 0xff4c4c4c;
    int64_t const SimulationSliderColor_Active = 0xff6c6c6c;
    int64_t const TextDecentColor = 0xff909090;
    int64_t const TextInfoColor = 0xff30b0b0;

    ImColor const MenuButton = ImColor::HSV(0.6f, 0.3f, 0.35f);
    ImColor const MenuButtonHovered = ImColor::HSV(0.6f, 1.0f, 1.0f);
    ImColor const MenuButtonActive = ImColor::HSV(0.6f, 0.6f, 0.6f);

    ImColor const ShutdownButton = ImColor::HSV(0.f, 0.6f, 0.6f);
    ImColor const ShutdownButtonHovered = ImColor::HSV(0.f, 1.0f, 1.0f);
    ImColor const ShutdownButtonActive = ImColor::HSV(0.f, 1.0f, 1.0f);

    float const WindowAlpha = 0.8f;
    float const SliderBarWidth = 40.0f;
}

class _StyleRepository
{
public:
    _StyleRepository();

    ImFont* getMediumFont() const;
    ImFont* getLargeFont() const;

    ImFont* getTestFont() const;

private:
    ImFont* _mediumFont = nullptr;
    ImFont* _largeFont = nullptr;
    ImFont* _testFont = nullptr;
};