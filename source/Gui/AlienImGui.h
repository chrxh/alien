#pragma once

#include "Base/Definitions.h"
#include "Definitions.h"

class AlienImGui
{
public:
    static void HelpMarker(std::string const& text);

    struct SliderFloatParameters
    {
        MEMBER_DECLARATION(SliderFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(SliderFloatParameters, float, min, 0);
        MEMBER_DECLARATION(SliderFloatParameters, float, max, 0);
        MEMBER_DECLARATION(SliderFloatParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(SliderFloatParameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(SliderFloatParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SliderFloatParameters, boost::optional<float>, defaultValue, boost::none);
        MEMBER_DECLARATION(SliderFloatParameters, boost::optional<std::string>, tooltip, boost::none);
    };
    static void SliderFloat(SliderFloatParameters const& parameters, float& value);

    struct SliderIntParameters
    {
        MEMBER_DECLARATION(SliderIntParameters, std::string, name, "");
        MEMBER_DECLARATION(SliderIntParameters, int, min, 0);
        MEMBER_DECLARATION(SliderIntParameters, int, max, 0);
        MEMBER_DECLARATION(SliderIntParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SliderIntParameters, boost::optional<int>, defaultValue, boost::none);
        MEMBER_DECLARATION(SliderIntParameters, boost::optional<std::string>, tooltip, boost::none);
    };
    static void SliderInt(SliderIntParameters const& parameters, int& value);

    struct SliderInputFloatParameters
    {
        MEMBER_DECLARATION(SliderInputFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(SliderInputFloatParameters, int, min, 0);
        MEMBER_DECLARATION(SliderInputFloatParameters, int, max, 0);
        MEMBER_DECLARATION(SliderInputFloatParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SliderInputFloatParameters, std::string, format, "%.3f");
    };
    static void SliderInputFloat(SliderInputFloatParameters const& parameters, float& value);

    struct InputIntParameters
    {
        MEMBER_DECLARATION(InputIntParameters, std::string, name, "");
        MEMBER_DECLARATION(InputIntParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputIntParameters, boost::optional<int>, defaultValue, boost::none);
        MEMBER_DECLARATION(InputIntParameters, boost::optional<std::string>, tooltip, boost::none);
    };
    static void InputInt(InputIntParameters const& parameters, int& value);

    struct InputFloatParameters{
        MEMBER_DECLARATION(InputFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloatParameters, float, step, 1.0f);
        MEMBER_DECLARATION(InputFloatParameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(InputFloatParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputFloatParameters, boost::optional<float>, defaultValue, boost::none);
        MEMBER_DECLARATION(InputFloatParameters, boost::optional<std::string>, tooltip, boost::none);
    };
    static void InputFloat(InputFloatParameters const& parameters, float& value);

    struct ComboParameters
    {
        MEMBER_DECLARATION(ComboParameters, std::string, name, "");
        MEMBER_DECLARATION(ComboParameters, int, textWidth, 100);
        MEMBER_DECLARATION(ComboParameters, int, defaultValue, 0);
        MEMBER_DECLARATION(ComboParameters, std::vector<std::string>, values, std::vector<std::string>());
    };
    static void Combo(ComboParameters const& parameters, int& value);

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
    static void Group(std::string const& text);
};
