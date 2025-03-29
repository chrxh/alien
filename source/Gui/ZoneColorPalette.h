#pragma once

#include <cstdint>

#include "EngineInterface/SimulationParametersTypes.h"

class ZoneColorPalette
{
public:
    ZoneColorPalette();

    FloatColorRGB getColor(int index) const;

    using Palette = FloatColorRGB[32];
    Palette& getReference();

private:
    FloatColorRGB _palette[32] = {};
};
