#pragma once

#include <cstdint>

#include "EngineInterface/SimulationParametersTypes.h"

class LayerColorPalette
{
public:
    LayerColorPalette();

    FloatColorRGB getColor(int index) const;

    using Palette = FloatColorRGB[32];
    Palette& getReference();

private:
    FloatColorRGB _palette[32] = {};
};
