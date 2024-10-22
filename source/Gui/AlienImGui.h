#pragma once

#include <functional>

#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/PreviewDescriptions.h"
#include "Definitions.h"
#include "EngineInterface/CellFunctionConstants.h"

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
        MEMBER_DECLARATION(SliderFloatParameters, float, textWidth, 100);
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
        MEMBER_DECLARATION(SliderIntParameters, float, textWidth, 100);
        MEMBER_DECLARATION(SliderIntParameters, bool, colorDependence, false);
        MEMBER_DECLARATION(SliderIntParameters, int const*, defaultValue, nullptr);
        MEMBER_DECLARATION(SliderIntParameters, int const*, disabledValue, nullptr);
        MEMBER_DECLARATION(SliderIntParameters, bool const*, defaultEnabledValue, nullptr);
        MEMBER_DECLARATION(SliderIntParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool SliderInt(SliderIntParameters const& parameters, int* value, bool* enabled = nullptr);

    struct SliderFloat2Parameters
    {
        MEMBER_DECLARATION(SliderFloat2Parameters, std::string, name, "");
        MEMBER_DECLARATION(SliderFloat2Parameters, RealVector2D, min, RealVector2D());
        MEMBER_DECLARATION(SliderFloat2Parameters, RealVector2D, max, RealVector2D());
        MEMBER_DECLARATION(SliderFloat2Parameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(SliderFloat2Parameters, float, textWidth, 100);
        MEMBER_DECLARATION(SliderFloat2Parameters, std::optional<RealVector2D>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(SliderFloat2Parameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(SliderFloat2Parameters, std::optional<std::function<bool(void)>>, getMousePickerEnabledFunc, std::nullopt);
        MEMBER_DECLARATION(SliderFloat2Parameters, std::optional<std::function<void(bool)>>, setMousePickerEnabledFunc, std::nullopt);
        MEMBER_DECLARATION(SliderFloat2Parameters, std::optional<std::function<std::optional<RealVector2D>(void)>>, getMousePickerPositionFunc, std::nullopt);
    };
    static bool SliderFloat2(SliderFloat2Parameters const& parameters, float& valueX, float& valueY);

    struct SliderInputFloatParameters
    {
        MEMBER_DECLARATION(SliderInputFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(SliderInputFloatParameters, float, min, 0);
        MEMBER_DECLARATION(SliderInputFloatParameters, float, max, 0);
        MEMBER_DECLARATION(SliderInputFloatParameters, float, textWidth, 100);
        MEMBER_DECLARATION(SliderInputFloatParameters, float, inputWidth, 50);
        MEMBER_DECLARATION(SliderInputFloatParameters, std::string, format, "%.3f");
    };
    static void SliderInputFloat(SliderInputFloatParameters const& parameters, float& value);

    struct InputIntParameters
    {
        MEMBER_DECLARATION(InputIntParameters, std::string, name, "");
        MEMBER_DECLARATION(InputIntParameters, float, textWidth, 100);
        MEMBER_DECLARATION(InputIntParameters, bool, keepTextEnabled, false);
        MEMBER_DECLARATION(InputIntParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputIntParameters, bool, infinity, false);
        MEMBER_DECLARATION(InputIntParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(InputIntParameters, std::optional<int>, disabledValue, std::nullopt);
        MEMBER_DECLARATION(InputIntParameters, bool, readOnly, false);
    };
    static bool InputInt(InputIntParameters const& parameters, int& value, bool* enabled = nullptr);
    static bool InputOptionalInt(InputIntParameters const& parameters, std::optional<int>& value);

    struct InputFloatParameters
    {
        MEMBER_DECLARATION(InputFloatParameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloatParameters, float, step, 1.0f);
        MEMBER_DECLARATION(InputFloatParameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(InputFloatParameters, float, textWidth, 100.0f);
        MEMBER_DECLARATION(InputFloatParameters, std::optional<float>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputFloatParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(InputFloatParameters, bool, readOnly, false);
    };
    static bool InputFloat(InputFloatParameters const& parameters, float& value);

    struct InputFloat2Parameters
    {
        MEMBER_DECLARATION(InputFloat2Parameters, std::string, name, "");
        MEMBER_DECLARATION(InputFloat2Parameters, std::string, format, "%.3f");
        MEMBER_DECLARATION(InputFloat2Parameters, float, textWidth, 100);
        MEMBER_DECLARATION(InputFloat2Parameters, std::optional<float>, defaultValue1, std::nullopt);
        MEMBER_DECLARATION(InputFloat2Parameters, std::optional<float>, defaultValue2, std::nullopt);
        MEMBER_DECLARATION(InputFloat2Parameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(InputFloat2Parameters, bool, readOnly, false);
    };
    static void InputFloat2(InputFloat2Parameters const& parameters, float& value1, float& value2);

    static bool ColorField(uint32_t cellColor, float width = 0, float height = 0);

    struct CheckboxColorMatrixParameters
    {
        MEMBER_DECLARATION(CheckboxColorMatrixParameters, std::string, name, "");
        MEMBER_DECLARATION(CheckboxColorMatrixParameters, float, textWidth, 100);
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
        MEMBER_DECLARATION(InputIntColorMatrixParameters, float, textWidth, 100);
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
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, float, textWidth, 100);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::optional<std::vector<std::vector<float>>>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(InputFloatColorMatrixParameters, std::optional<std::vector<std::vector<float>>>, disabledValue, std::nullopt);
    };
    static void InputFloatColorMatrix(InputFloatColorMatrixParameters const& parameters, float (&value)[MAX_COLORS][MAX_COLORS], bool* enabled = nullptr);

    struct InputTextParameters
    {
        MEMBER_DECLARATION(InputTextParameters, std::string, name, "");
        MEMBER_DECLARATION(InputTextParameters, std::string, hint, "");
        MEMBER_DECLARATION(InputTextParameters, float, width, 0);
        MEMBER_DECLARATION(InputTextParameters, float, textWidth, 100);
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
        MEMBER_DECLARATION(InputTextMultilineParameters, float, textWidth, 100);
        MEMBER_DECLARATION(InputTextMultilineParameters, float, height, 100.0f);
    };
    static void InputTextMultiline(InputTextMultilineParameters const& parameters, std::string& text);

    struct ComboParameters
    {
        MEMBER_DECLARATION(ComboParameters, std::string, name, "");
        MEMBER_DECLARATION(ComboParameters, float, textWidth, 100);
        MEMBER_DECLARATION(ComboParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(ComboParameters, std::vector<std::string>, values, std::vector<std::string>());
        MEMBER_DECLARATION(ComboParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Combo(ComboParameters& parameters, int& value, bool* enabled = nullptr);

    struct SwitcherParameters
    {
        MEMBER_DECLARATION(SwitcherParameters, std::string, name, "");
        MEMBER_DECLARATION(SwitcherParameters, float, width, 0);
        MEMBER_DECLARATION(SwitcherParameters, float, textWidth, 100);
        MEMBER_DECLARATION(SwitcherParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(SwitcherParameters, std::vector<std::string>, values, std::vector<std::string>());
        MEMBER_DECLARATION(SwitcherParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Switcher(SwitcherParameters& parameters, int& value, bool* enabled = nullptr);

    struct ComboColorParameters
    {
        MEMBER_DECLARATION(ComboColorParameters, std::string, name, "");
        MEMBER_DECLARATION(ComboColorParameters, float, width, 0);
        MEMBER_DECLARATION(ComboColorParameters, float, textWidth, 100);
        MEMBER_DECLARATION(ComboColorParameters, bool, keepTextEnabled, false);
        MEMBER_DECLARATION(ComboColorParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(ComboColorParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool ComboColor(ComboColorParameters const& parameters, int& value, bool* enabled = nullptr);
    static bool ComboOptionalColor(ComboColorParameters const& parameters, std::optional<int>& value);

    struct InputColorTransitionParameters
    {
        MEMBER_DECLARATION(InputColorTransitionParameters, std::string, name, "");
        MEMBER_DECLARATION(InputColorTransitionParameters, int, color, 0);
        MEMBER_DECLARATION(InputColorTransitionParameters, float, textWidth, 100);
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
        MEMBER_DECLARATION(CheckboxParameters, float, textWidth, 100);
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

    struct SelectableButtonParameters 
    {
        MEMBER_DECLARATION(SelectableButtonParameters, std::string, name, "");
        MEMBER_DECLARATION(SelectableButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(SelectableButtonParameters, float, width, 0);
    };
    static bool SelectableButton(SelectableButtonParameters const& parameters, bool& value);

    static void Text(std::string const& text);
    static void BoldText(std::string const& text);
    static void MonospaceText(std::string const& text);

    static void BeginMenuBar();
    static void BeginMenu(std::string const& text, bool& toggled, float focus = true);
    struct MenuItemParameters
    {
        MEMBER_DECLARATION(MenuItemParameters, std::string, name, "");
        MEMBER_DECLARATION(MenuItemParameters, bool, keyCtrl, false);
        MEMBER_DECLARATION(MenuItemParameters, bool, keyAlt, false);
        MEMBER_DECLARATION(MenuItemParameters, std::optional<ImGuiKey>, key, std::nullopt);
        MEMBER_DECLARATION(MenuItemParameters, bool, disabled, false);
        MEMBER_DECLARATION(MenuItemParameters, bool, selected, false);
        MEMBER_DECLARATION(MenuItemParameters, bool, closeMenuWhenItemClicked, true);
    };
    static void MenuItem(MenuItemParameters const& parameters, std::function<void()> const& action);
    static void MenuSeparator();
    static void EndMenu();
    static void MenuShutdownButton(std::function<void()> const& action);
    static void EndMenuBar();

    struct ColorButtonWithPickerParameters
    {
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, std::string, name, "");
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, float, textWidth, 100);
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, std::optional<uint32_t>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(ColorButtonWithPickerParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void ColorButtonWithPicker(ColorButtonWithPickerParameters const& parameters, uint32_t& color, uint32_t& backupColor, uint32_t (&savedPalette)[32]);

    static void NegativeSpacing();
    static void Separator();
    static void MovableSeparator(float& height);

    static void Group(std::string const& text, std::optional<std::string> const& tooltip = std::nullopt);

    static bool ToolbarButton(std::string const& text);
    static bool SelectableToolbarButton(std::string const& text, int& value, int selectionValue, int deselectionValue);

    static void VerticalSeparator(float length = 23.0f);
    static void ToolbarSeparator();
    static bool Button(std::string const& text, float size = 0);
    static bool CollapseButton(bool collapsed);

    struct TreeNodeParameters
    {
        MEMBER_DECLARATION(TreeNodeParameters, std::string, text, "");
        MEMBER_DECLARATION(TreeNodeParameters, bool, highlighted, false);
        MEMBER_DECLARATION(TreeNodeParameters, bool, defaultOpen, true);
    };
    static bool BeginTreeNode(TreeNodeParameters const& parameters);
    static void EndTreeNode();

    struct ButtonParameters
    {
        MEMBER_DECLARATION(ButtonParameters, std::string, buttonText, "");
        MEMBER_DECLARATION(ButtonParameters, std::string, name, "");
        MEMBER_DECLARATION(ButtonParameters, float, textWidth, 100);
        MEMBER_DECLARATION(ButtonParameters, bool, showDisabledRevertButton, false);
        MEMBER_DECLARATION(ButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Button(ButtonParameters const& parameters);

    struct SpinnerParameters
    {
        MEMBER_DECLARATION(SpinnerParameters, RealVector2D, pos, RealVector2D());
    };
    static void Spinner(SpinnerParameters const& parameters);

    static void Tooltip(std::string const& text, bool delay = true);
    static void Tooltip(std::function<std::string()> const& textFunc, bool delay = true);

    static void ConvertRGBtoHSV(uint32_t rgb, float& h, float& s, float& v);

    static bool ShowPreviewDescription(PreviewDescription const& desc, float& zoom, std::optional<int>& selectedNode);

    struct CellFunctionComboParameters
    {
        MEMBER_DECLARATION(CellFunctionComboParameters, std::string, name, "");
        MEMBER_DECLARATION(CellFunctionComboParameters, float, textWidth, 100);
        MEMBER_DECLARATION(CellFunctionComboParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool CellFunctionCombo(CellFunctionComboParameters& parameters, int& value);
    struct AngleAlignmentComboParameters
    {
        MEMBER_DECLARATION(AngleAlignmentComboParameters, std::string, name, "");
        MEMBER_DECLARATION(AngleAlignmentComboParameters, float, textWidth, 100);
        MEMBER_DECLARATION(AngleAlignmentComboParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool AngleAlignmentCombo(AngleAlignmentComboParameters& parameters, int& value);

    struct NeuronSelectionParameters
    {
        MEMBER_DECLARATION(NeuronSelectionParameters, std::string, name, "");
        MEMBER_DECLARATION(NeuronSelectionParameters, float, step, 0.05f);
        MEMBER_DECLARATION(NeuronSelectionParameters, std::string, format, "%.2f");
        MEMBER_DECLARATION(NeuronSelectionParameters, float, rightMargin, 0);
    };
    static void NeuronSelection(
        NeuronSelectionParameters const& parameters,
        std::vector<std::vector<float>>& weights,
        std::vector<float>& biases,
        std::vector<NeuronActivationFunction>& activationFunctions);

    static void OnlineSymbol();
    static void LastDayOnlineSymbol();

    class DynamicTableLayout
    {
    public:
        static int calcNumColumns(float tableWidth, float columnWidth);

        DynamicTableLayout(float columnWidth);

        bool begin();
        void end();
        void next();

    private:
        float _columnWidth = 0;
        int _numColumns = 0;
        int _elementNumber = 0;
    };

    static void RotateStart(ImDrawList* drawList);
    static void RotateEnd(float angle, ImDrawList* drawList);

private:

    template <typename Parameter, typename T>
    static bool BasicSlider(Parameter const& parameters, T* value, bool* enabled);


    template<typename T>
    struct BasicInputColorMatrixParameters
    {
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, std::string, name, "");
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, T, min, static_cast<T>(0));
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, T, max, static_cast<T>(0));
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, bool, logarithmic, false);
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, std::string, format, "%.2f");
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, float, textWidth, 100);
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, std::optional<std::vector<std::vector<T>>>, defaultValue, std::nullopt);
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER_DECLARATION(BasicInputColorMatrixParameters, std::optional<std::vector<std::vector<T>>>, disabledValue, std::nullopt);
    };
    template <typename T>
    static void BasicInputColorMatrix(BasicInputColorMatrixParameters<T> const& parameters, T (&value)[MAX_COLORS][MAX_COLORS], bool* enabled = nullptr);

    static ImVec2 RotationCenter(ImDrawList* drawList);

    static bool revertButton(std::string const& id);

private:
    static std::unordered_set<unsigned int> _basicSilderExpanded;

    static int _rotationStartIndex;

    static std::unordered_map<unsigned int, int> _neuronSelectedInput;
    static std::unordered_map<unsigned int, int> _neuronSelectedOutput;

    static bool _menuBarVisible;
    static bool* _menuButtonToggled;
    static bool _menuButtonToToggle;
};
