#pragma once

#include <cstdint>

class ZoneColorPalette
{
public:
    ZoneColorPalette();

    uint32_t getColor(int index) const;

    using Palette = uint32_t[32];
    Palette& getReference();

private:
    uint32_t _palette[32] = {};
};
