#pragma once

#include "Base/Definitions.h"
#include "Definitions.h"

class AlienImGui
{
public:
    static void HelpMarker(std::string const& text);
    static bool BeginMenuButton(std::string const& text, bool& toggle,
                                std::string const& popup);  //return toggle
    static void EndMenuButton();
    static bool ShutdownButton();
    static void ColorButtonWithPicker(
        std::string const& text,
        uint32_t& color,
        uint32_t& backupColor,
        uint32_t (&savedPalette)[32],
        RealVector2D const& size);
};
