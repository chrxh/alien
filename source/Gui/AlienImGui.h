#pragma once

#include "Base/Definitions.h"
#include "Definitions.h"

class AlienImGui
{
public:
    static void HelpMarker(std::string const& text);
    static void SliderFloat(
        std::string const& name,
        float& value,
        float defaultValue,
        float min,
        float max,
        bool logarithmic = false,
        std::string const& format = "%.3f",
        boost::optional<std::string> tooltip = boost::none);
    static void SliderInt(
        std::string const& name,
        int& value,
        int defaultValue,
        int min,
        int max,
        boost::optional<std::string> tooltip = boost::none);
    static void InputInt(
        std::string const& name,
        int& value,
        int defaultValue,
        boost::optional<std::string> tooltip = boost::none);
    static void Combo(std::string const& name, int& value, int defaultValue, std::vector<std::string> const& values);
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

    static void Separator();
};
