#pragma once

#include <functional>

#include "Base/Definitions.h"
#include "EngineInterface/FundamentalConstants.h"
#include "EngineInterface/PreviewDescriptions.h"
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
        MEMBER_DECLARATION(SliderFloatParameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(SliderFloatParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(SliderFloatParameters, bool, infinity, false);
        MEMBER_DECLARATION(SliderFloatParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SliderFloatParameters, bool, colorDependence, false);
        MEMBER_DECLARATION(SliderFloatParameters, float const*, defaultValue, nullptr);
        MEMBER_DECLARATION(SliderFloatParameters, float const*, disabledValue, nullptr);
        MEMBER_DECLARATION(SliderFloatParameters, bool const*, defaultEnabledValue, nullptr);
        MEMBER_DECLARATION(SliderFloatParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool SliderFloat(SliderFloatParameters const& parameters, float* value, bool* enabled = nullptr);

    struct SliderIntParameters
    {
        MEMBER_DECLARATION(SliderIntParameters, std::string, name, "");
        MEMBER_DECLARATION(SliderIntParameters, int, min, 0);
        MEMBER_DECLARATION(SliderIntParameters, int, max, 0);
        MEMBER_DECLARATION(SliderIntParameters, std::string, format, "%d");
        MEMBER_DECLARATION(SliderIntParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(SliderIntParameters, bool, infinity, false);
        MEMBER_DECLARATION(SliderIntParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SliderIntParameters, bool, colorDependence, false);
        MEMBER_DECLARATION(SliderIntParameters, int const*, defaultValue, nullptr);
        MEMBER_DECLARATION(SliderIntParameters, int const*, disabledValue, nullptr);
        MEMBER_DECLARATION(SliderIntParameters, bool const*, defaultEnabledValue, nullptr);
        MEMBER_DECLARATION(SliderIntParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool SliderInt(SliderIntParameters const& parameters, int* value, bool* enabled = nullptr);

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
        MEMBER_DECLARATION(InputIntParameters, std::optional<int>, disabledValue, std::nullopt);
    };
    static bool InputInt(InputIntParameters const& parameters, int& value, bool* enabled = nullptr);
    static bool InputOptionalInt(InputIntParameters const& parameters, std::optional<int>& value);

    struct InputFloatParameters
    {
        MEMBER_DECLARATION(InputFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloatParameters, float, step, 1.0f);
        MEMBER_DECLARATION(InputFloatParameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(InputFloatParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputFloatParameters, std::optional<float>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputFloatParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(InputFloatParameters, bool, readOnly, false);
    };
    static bool InputFloat(InputFloatParameters const& parameters, float& value);

    struct InputFloat2Parameters
    {
        MEMBER_DECLARATION(InputFloat2Parameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloat2Parameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(InputFloat2Parameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputFloat2Parameters, std::optional<float>, defaultValue1, std::nullopt);
        MEMBER_DECLARATION(InputFloat2Parameters, std::optional<float>, defaultValue2, std::nullopt);
        MEMBER_DECLARATION(InputFloat2Parameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(InputFloat2Parameters, bool, readOnly, false);
    };
    static void InputFloat2(InputFloat2Parameters const& parameters, float& value1, float& value2);

    static bool ColorField(uint32_t cellColor, int width = 0);

    struct CheckboxColorMatrixParameters
    {
        MEMBER_DECLARATION(CheckboxColorMatrixParameters, std::string, name, "");
        MEMBER_DECLARATION(CheckboxColorMatrixParameters, int, textWidth, 100);
        MEMBER_DECLARATION(CheckboxColorMatrixParameters, std::optional<std::vector<std::vector<bool>>>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(CheckboxColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void CheckboxColorMatrix(CheckboxColorMatrixParameters const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);

    struct InputIntColorMatrixParameters
    {
        MEMBER_DECLARATION(InputIntColorMatrixParameters, std::string, name, "");
        MEMBER_DECLARATION(InputIntColorMatrixParameters, int, min, 0);
        MEMBER_DECLARATION(InputIntColorMatrixParameters, int, max, 0);
        MEMBER_DECLARATION(InputIntColorMatrixParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(InputIntColorMatrixParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputIntColorMatrixParameters, std::optional<std::vector<std::vector<int>>>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputIntColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void InputIntColorMatrix(InputIntColorMatrixParameters const& parameters, int (&value)[MAX_COLORS][MAX_COLORS]);

    struct InputFloatColorMatrixParameters
    {
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, float, min, 0);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, float, max, 0);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::string, format, "%.2f");
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::optional<std::vector<std::vector<float>>>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void InputFloatColorMatrix(InputFloatColorMatrixParameters const& parameters, float (&value)[MAX_COLORS][MAX_COLORS]);

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
        MEMBER_DECLARATION(ComboParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Combo(ComboParameters& parameters, int& value);

    struct SwitcherParameters
    {
        MEMBER_DECLARATION(SwitcherParameters, std::string, name, "");
        MEMBER_DECLARATION(SwitcherParameters, int, textWidth, 100);
        MEMBER_DECLARATION(SwitcherParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(SwitcherParameters, std::vector<std::string>, values, std::vector<std::string>());
        MEMBER_DECLARATION(SwitcherParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Switcher(SwitcherParameters& parameters, int& value);

    struct ComboColorParameters
    {
        MEMBER_DECLARATION(ComboColorParameters, std::string, name, "");
        MEMBER_DECLARATION(ComboColorParameters, int, textWidth, 100);
        MEMBER_DECLARATION(ComboColorParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(ComboColorParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool ComboColor(ComboColorParameters const& parameters, int& value);

    struct InputColorTransitionParameters
    {
        MEMBER_DECLARATION(InputColorTransitionParameters, std::string, name, "");
        MEMBER_DECLARATION(InputColorTransitionParameters, int, color, 0);
        MEMBER_DECLARATION(InputColorTransitionParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputColorTransitionParameters, int, min, 1);
        MEMBER_DECLARATION(InputColorTransitionParameters, int, max, 10000000);
        MEMBER_DECLARATION(InputColorTransitionParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(InputColorTransitionParameters, bool, infinity, false);
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
    static void BoldText(std::string const& text);
    static void MonospaceText(std::string const& text);

    static bool BeginMenuButton(std::string const& text, bool& toggle, std::string const& popup, float focus = true);  //return toggle
    static void EndMenuButton();
    static bool ShutdownButton();

    struct ColorButtonWithPickerParameters
    {
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, std::string, name, "");
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, int, textWidth, 100);
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, std::optional<uint32_t>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void ColorButtonWithPicker(ColorButtonWithPickerParameters const& parameters, uint32_t& color, uint32_t& backupColor, uint32_t (&savedPalette)[32]);

    static void NegativeSpacing();
    static void Separator();
    static void Group(std::string const& text);

    static bool ToolbarButton(std::string const& text);
    static void VerticalSeparator(float length = 23.0f);
    static void ToolbarSeparator();
    static bool Button(std::string const& text, float size = 0);

    struct ButtonParameters
    {
        MEMBER_DECLARATION(ButtonParameters, std::string, buttonText, "");
        MEMBER_DECLARATION(ButtonParameters, std::string, name, "");
        MEMBER_DECLARATION(ButtonParameters, int, textWidth, 100);
        MEMBER_DECLARATION(ButtonParameters, bool, showDisabledRevertButton, false);
        MEMBER_DECLARATION(ButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Button(ButtonParameters const& parameters);

    static void Tooltip(std::string const& text, bool delay = true);
    static void Tooltip(std::function<std::string()> const& textFunc, bool delay = true);

    static void ConvertRGBtoHSV(uint32_t rgb, float& h, float& s, float& v);

    static bool ShowPreviewDescription(PreviewDescription const& desc, float& zoom, std::optional<int>& selectedNode);

    struct CellFunctionComboParameters
    {
        MEMBER_DECLARATION(CellFunctionComboParameters, std::string, name, "");
        MEMBER_DECLARATION(CellFunctionComboParameters, int, textWidth, 100);
        MEMBER_DECLARATION(CellFunctionComboParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool CellFunctionCombo(CellFunctionComboParameters& parameters, int& value);
    struct AngleAlignmentComboParameters
    {
        MEMBER_DECLARATION(AngleAlignmentComboParameters, std::string, name, "");
        MEMBER_DECLARATION(AngleAlignmentComboParameters, int, textWidth, 100);
        MEMBER_DECLARATION(AngleAlignmentComboParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool AngleAlignmentCombo(AngleAlignmentComboParameters& parameters, int& value);

    struct NeuronSelectionParameters
    {
        MEMBER_DECLARATION(NeuronSelectionParameters, std::string, name, "");
        MEMBER_DECLARATION(NeuronSelectionParameters, float, step, 0.05f);
        MEMBER_DECLARATION(NeuronSelectionParameters, std::string, format, "%.2f");
        MEMBER_DECLARATION(NeuronSelectionParameters, int, outputButtonPositionFromRight, 0);
    };
    static void NeuronSelection(
        NeuronSelectionParameters const& parameters,
        std::vector<std::vector<float>> const& weights,
        std::vector<float> const& biases,
        int& selectedInput,
        int& selectedOutput
    );

    static void OnlineSymbol();

private:

    template <typename Parameter, typename T>
    static bool BasicSlider(Parameter const& parameters, T* value, bool* enabled);


    template<typename T>
    struct BasicInputColorMatrixParameters
    {
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, T, min, static_cast<T>(0));
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, T, max, static_cast<T>(0));
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::string, format, "%.2f");
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, int, textWidth, 100);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::optional<std::vector<std::vector<T>>>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    template <typename T>
    static void BasicInputColorMatrix(BasicInputColorMatrixParameters<T> const& parameters, T (&value)[MAX_COLORS][MAX_COLORS]);

    static void RotateStart();
    static ImVec2 RotationCenter();
    static void RotateEnd(float angle);

    static std::unordered_set<unsigned int> _isExpanded;
    static int _rotationStartIndex;
};
