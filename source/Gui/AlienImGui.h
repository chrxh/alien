#pragma once

#include "Definitions.h"

class AlienImGui
{
public:
    static void HelpMarker(std::string const& text);
    static bool BeginMenuButton(std::string const& text, bool& toggle,
                                std::string const& popup);  //return toggle
    static void EndMenuButton();
};
