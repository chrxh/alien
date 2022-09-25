#pragma once

#include <functional>

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
        MEMBER_DECLARATION(SliderFloatParameters, std::optional<float>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(SliderFloatParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool SliderFloat(SliderFloatParameters const& parameters, float& value);

    struct SliderIntParameters
    {
        MEMBER_DECLARATION(SliderIntParameters, std::string, name, "");
        MEMBER_DECLARATION(SliderIntParameters, int, min, 0);
        MEMBER_DECLARATION(SliderIntParameters, int, max, 0);
        MEMBER_DECLARATION(SliderIntParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(SliderIntParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SliderIntParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(SliderIntParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool SliderInt(SliderIntParameters const& parameters, int& value);

    struct SliderInputFloatParameters
    {
        MEMBER_DECLARATION(SliderInputFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(SliderInputFloatParameters, float, min, 0);
        MEMBER_DECLARATION(SliderInputFloatParameters, float, max, 0);
        MEMBER_DECLARATION(SliderInputFloatParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SliderInputFloatParameters, int, inputWidth, 50);
        MEMBER_DECLARATION(SliderInputFloatParameters, std::string, format, "%.3f");
    };
    static void SliderInputFloat(SliderInputFloatParameters const& parameters, float& value);

    struct InputIntParameters
    {
        MEMBER_DECLARATION(InputIntParameters, std::string, name, "");
        MEMBER_DECLARATION(InputIntParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputIntParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputIntParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool InputInt(InputIntParameters const& parameters, int& value);

    struct InputFloatParameters
    {
        MEMBER_DECLARATION(InputFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloatParameters, float, step, 1.0f);
        MEMBER_DECLARATION(InputFloatParameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(InputFloatParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputFloatParameters, std::optional<float>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputFloatParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void InputFloat(InputFloatParameters const& parameters, float& value);

    static void ColorField(uint32_t cellColor, int width = 0);

    struct InputMatrixParameters
    {
        MEMBER_DECLARATION(InputMatrixParameters, std::string, name, "");
        MEMBER_DECLARATION(InputMatrixParameters, std::string, format, "%.2f");
        MEMBER_DECLARATION(InputMatrixParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputMatrixParameters, std::optional<std::vector<std::vector<float>>>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void InputColorMatrix(InputMatrixParameters const& parameters, float (&value)[7][7]);

    struct InputTextParameters
    {
        MEMBER_DECLARATION(InputTextParameters, std::string, name, "");
        MEMBER_DECLARATION(InputTextParameters, std::string, hint, "");
        MEMBER_DECLARATION(InputTextParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputTextParameters, bool, monospaceFont, false);
        MEMBER_DECLARATION(InputTextParameters, bool, readOnly, false);
        MEMBER_DECLARATION(InputTextParameters, bool, password, false);
        MEMBER_DECLARATION(InputTextParameters, std::optional<std::string>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputTextParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool InputText(InputTextParameters const& parameters, char* buffer, int bufferSize);
    static bool InputText(InputTextParameters const& parameters, std::string& text);

    struct InputTextMultilineParameters
    {
        MEMBER_DECLARATION(InputTextMultilineParameters, std::string, name, "");
        MEMBER_DECLARATION(InputTextMultilineParameters, std::string, hint, "");
        MEMBER_DECLARATION(InputTextMultilineParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputTextMultilineParameters, float, height, 100.0f);
    };
    static void InputTextMultiline(InputTextMultilineParameters const& parameters, std::string& text);

    struct ComboParameters
    {
        MEMBER_DECLARATION(ComboParameters, std::string, name, "");
        MEMBER_DECLARATION(ComboParameters, int, textWidth, 100);
        MEMBER_DECLARATION(ComboParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(ComboParameters, std::vector<std::string>, values, std::vector<std::string>());
    };
    static bool Combo(ComboParameters& parameters, int& value);

    struct ComboColorParameters
    {
        MEMBER_DECLARATION(ComboColorParameters, std::string, name, "");
        MEMBER_DECLARATION(ComboColorParameters, std::optional<int>, defaultValue, std::nullopt);
    };
    static bool ComboColor(ComboColorParameters const& parameters, int& value);

    struct InputColorTransitionParameters
    {
        MEMBER_DECLARATION(InputColorTransitionParameters, std::string, name, "");
        MEMBER_DECLARATION(InputColorTransitionParameters, int, color, 0);
        MEMBER_DECLARATION(InputColorTransitionParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputColorTransitionParameters, int, min, 0);
        MEMBER_DECLARATION(InputColorTransitionParameters, int, max, 1000000);
        MEMBER_DECLARATION(InputColorTransitionParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(InputColorTransitionParameters, std::optional<int>, defaultTargetColor, std::nullopt);
        MEMBER_DECLARATION(InputColorTransitionParameters, std::optional<int>, defaultTransitionAge, std::nullopt);
        MEMBER_DECLARATION(InputColorTransitionParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void InputColorTransition(InputColorTransitionParameters const& parameters, int sourceColor, int& targetColor, int& transitionAge);

    struct CheckboxParameters
    {
        MEMBER_DECLARATION(CheckboxParameters, std::string, name, "");
        MEMBER_DECLARATION(CheckboxParameters, int, textWidth, 100);
        MEMBER_DECLARATION(CheckboxParameters, std::optional<bool>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(CheckboxParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Checkbox(CheckboxParameters const& parameters, bool& value);

    struct ToggleButtonParameters
    {
        MEMBER_DECLARATION(ToggleButtonParameters, std::string, name, "");
        MEMBER_DECLARATION(ToggleButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool ToggleButton(ToggleButtonParameters const& parameters, bool& value);

    static void Text(std::string const& text);

    static bool BeginMenuButton(std::string const& text, bool& toggle, std::string const& popup, float focus = true);  //return toggle
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

    static bool ToolbarButton(std::string const& text);
    static bool Button(std::string const& text);

    static void Tooltip(std::string const& text);
    static void Tooltip(std::function<std::string()> const& textFunc);

    static void convertRGBtoHSV(uint32_t rgb, float& h, float& s, float& v);
};
