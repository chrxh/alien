#pragma once

#include "Definitions.h"

#include <cstdint>

namespace Const
{
    int64_t const SimulationSliderColor_Base = 0xff4c4c4c;
    int64_t const SimulationSliderColor_Active = 0xff6c6c6c;
    int64_t const TextDecentColor = 0xff909090;
    int64_t const TextInfoColor = 0xff30b0b0;
    float const WindowAlpha = 0.8f;
    float const SliderBarWidth = 40.0f;
}

class _StyleRepository
{
public:
    _StyleRepository();

    ImFont* getMediumFont() const;
    ImFont* getLargeFont() const;

private:
    ImFont* _mediumFont = nullptr;
    ImFont* _largeFont = nullptr;
};