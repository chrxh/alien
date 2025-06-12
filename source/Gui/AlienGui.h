#pragma once

#include <chrono>
#include <functional>

#include <imgui.h>

#include "Base/Definitions.h"
#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/PreviewDescriptions.h"
#include "EngineInterface/CellTypeConstants.h"
#include "EngineInterface/SimulationParametersTypes.h"

#include "Definitions.h"

struct FilterStackElement
{
    std::string text;
    bool alreadyMatched = false;
};

struct TreeNodeStackElement
{
    float treeNodeStartCursorPosY = 0;
    ImGuiID treeNodeId = 0;
    bool isOpen = false;
};

struct TreeNodeInfo
{
    std::chrono::steady_clock::time_point invisibleTimepoint;
    bool isEmpty = false;
};

class AlienGui
{
public:
    static void HelpMarker(std::string const& text);

    struct SliderFloatParameters
    {
        MEMBER(SliderFloatParameters, std::string, name, "");
        MEMBER(SliderFloatParameters, float, min, 0);
        MEMBER(SliderFloatParameters, float, max, 0);
        MEMBER(SliderFloatParameters, std::string, format, "%.3f");
        MEMBER(SliderFloatParameters, bool, logarithmic, false);
        MEMBER(SliderFloatParameters, bool, infinity, false);
        MEMBER(SliderFloatParameters, float, textWidth, 100);
        MEMBER(SliderFloatParameters, bool, colorDependence, false);
        MEMBER(SliderFloatParameters, bool, readOnly, false);
        MEMBER(SliderFloatParameters, float const*, defaultValue, nullptr);
        MEMBER(SliderFloatParameters, float const*, disabledValue, nullptr);
        MEMBER(SliderFloatParameters, bool const*, defaultEnabledValue, nullptr);
        MEMBER(SliderFloatParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool SliderFloat(SliderFloatParameters const& parameters, float* value, bool* enabled = nullptr, bool* pinned = nullptr);

    struct SliderIntParameters
    {
        MEMBER(SliderIntParameters, std::string, name, "");
        MEMBER(SliderIntParameters, int, min, 0);
        MEMBER(SliderIntParameters, int, max, 0);
        MEMBER(SliderIntParameters, std::string, format, "%d");
        MEMBER(SliderIntParameters, bool, logarithmic, false);
        MEMBER(SliderIntParameters, bool, infinity, false);
        MEMBER(SliderIntParameters, float, textWidth, 100);
        MEMBER(SliderIntParameters, bool, colorDependence, false);
        MEMBER(SliderIntParameters, bool, readOnly, false);
        MEMBER(SliderIntParameters, int const*, defaultValue, nullptr);
        MEMBER(SliderIntParameters, int const*, disabledValue, nullptr);
        MEMBER(SliderIntParameters, bool const*, defaultEnabledValue, nullptr);
        MEMBER(SliderIntParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool SliderInt(SliderIntParameters const& parameters, int* value, bool* enabled = nullptr, bool* pinned = nullptr);

    struct SliderFloat2Parameters
    {
        MEMBER(SliderFloat2Parameters, std::string, name, "");
        MEMBER(SliderFloat2Parameters, RealVector2D, min, RealVector2D());
        MEMBER(SliderFloat2Parameters, RealVector2D, max, RealVector2D());
        MEMBER(SliderFloat2Parameters, std::string, format, "%.3f");
        MEMBER(SliderFloat2Parameters, float, textWidth, 100);
        MEMBER(SliderFloat2Parameters, std::optional<RealVector2D>, defaultValue, std::nullopt);
        MEMBER(SliderFloat2Parameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER(SliderFloat2Parameters, std::optional<std::function<bool(void)>>, getMousePickerEnabledFunc, std::nullopt);
        MEMBER(SliderFloat2Parameters, std::optional<std::function<void(bool)>>, setMousePickerEnabledFunc, std::nullopt);
        MEMBER(SliderFloat2Parameters, std::optional<std::function<std::optional<RealVector2D>(void)>>, getMousePickerPositionFunc, std::nullopt);
    };
    static bool SliderFloat2(SliderFloat2Parameters const& parameters, float& valueX, float& valueY);

    struct SliderInputFloatParameters
    {
        MEMBER(SliderInputFloatParameters, std::string, name, "");
        MEMBER(SliderInputFloatParameters, float, min, 0);
        MEMBER(SliderInputFloatParameters, float, max, 0);
        MEMBER(SliderInputFloatParameters, float, textWidth, 100);
        MEMBER(SliderInputFloatParameters, float, inputWidth, 50);
        MEMBER(SliderInputFloatParameters, std::string, format, "%.3f");
    };
    static void SliderInputFloat(SliderInputFloatParameters const& parameters, float& value);

    struct InputIntParameters
    {
        MEMBER(InputIntParameters, std::string, name, "");
        MEMBER(InputIntParameters, float, textWidth, 100);
        MEMBER(InputIntParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER(InputIntParameters, bool, infinity, false);
        MEMBER(InputIntParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER(InputIntParameters, std::optional<int>, disabledValue, std::nullopt);
        MEMBER(InputIntParameters, bool, readOnly, false);
    };
    static bool InputInt(InputIntParameters const& parameters, int& value, bool* enabled = nullptr);
    static bool InputOptionalInt(InputIntParameters const& parameters, std::optional<int>& value);

    struct InputFloatParameters
    {
        MEMBER(InputFloatParameters, std::string, name, "");
        MEMBER(InputFloatParameters, float, step, 1.0f);
        MEMBER(InputFloatParameters, std::string, format, "%.3f");
        MEMBER(InputFloatParameters, float, textWidth, 100.0f);
        MEMBER(InputFloatParameters, std::optional<float>, defaultValue, std::nullopt);
        MEMBER(InputFloatParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER(InputFloatParameters, bool, readOnly, false);
    };
    static bool InputFloat(InputFloatParameters const& parameters, float& value);

    struct InputFloat2Parameters
    {
        MEMBER(InputFloat2Parameters, std::string, name, "");
        MEMBER(InputFloat2Parameters, std::string, format, "%.3f");
        MEMBER(InputFloat2Parameters, float, textWidth, 100);
        MEMBER(InputFloat2Parameters, std::optional<float>, defaultValue1, std::nullopt);
        MEMBER(InputFloat2Parameters, std::optional<float>, defaultValue2, std::nullopt);
        MEMBER(InputFloat2Parameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER(InputFloat2Parameters, bool, readOnly, false);
    };
    static void InputFloat2(InputFloat2Parameters const& parameters, float& value1, float& value2);

    static bool ColorField(uint32_t cellColor, float width = 0, float height = 0);

    struct CheckboxColorMatrixParameters
    {
        MEMBER(CheckboxColorMatrixParameters, std::string, name, "");
        MEMBER(CheckboxColorMatrixParameters, float, textWidth, 100);
        MEMBER(CheckboxColorMatrixParameters, std::optional<std::vector<std::vector<bool>>>, defaultValue, std::nullopt);
        MEMBER(CheckboxColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void CheckboxColorMatrix(CheckboxColorMatrixParameters const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS]);

    struct InputIntColorMatrixParameters
    {
        MEMBER(InputIntColorMatrixParameters, std::string, name, "");
        MEMBER(InputIntColorMatrixParameters, int, min, 0);
        MEMBER(InputIntColorMatrixParameters, int, max, 0);
        MEMBER(InputIntColorMatrixParameters, bool, logarithmic, false);
        MEMBER(InputIntColorMatrixParameters, float, textWidth, 100);
        MEMBER(InputIntColorMatrixParameters, std::optional<std::vector<std::vector<int>>>, defaultValue, std::nullopt);
        MEMBER(InputIntColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void InputIntColorMatrix(InputIntColorMatrixParameters const& parameters, int (&value)[MAX_COLORS][MAX_COLORS]);

    struct InputFloatColorMatrixParameters
    {
        MEMBER(InputFloatColorMatrixParameters, std::string, name, "");
        MEMBER(InputFloatColorMatrixParameters, float, min, 0);
        MEMBER(InputFloatColorMatrixParameters, float, max, 0);
        MEMBER(InputFloatColorMatrixParameters, bool, logarithmic, false);
        MEMBER(InputFloatColorMatrixParameters, std::string, format, "%.2f");
        MEMBER(InputFloatColorMatrixParameters, float, textWidth, 100);
        MEMBER(InputFloatColorMatrixParameters, std::optional<std::vector<std::vector<float>>>, defaultValue, std::nullopt);
        MEMBER(InputFloatColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER(InputFloatColorMatrixParameters, std::optional<std::vector<std::vector<float>>>, disabledValue, std::nullopt);
    };
    static void InputFloatColorMatrix(InputFloatColorMatrixParameters const& parameters, float (&value)[MAX_COLORS][MAX_COLORS], bool* enabled = nullptr);

    struct InputTextParameters
    {
        MEMBER(InputTextParameters, std::string, name, "");
        MEMBER(InputTextParameters, std::string, hint, "");
        MEMBER(InputTextParameters, float, width, 0);
        MEMBER(InputTextParameters, float, textWidth, 100);
        MEMBER(InputTextParameters, bool, monospaceFont, false);
        MEMBER(InputTextParameters, bool, bold, false);
        MEMBER(InputTextParameters, bool, readOnly, false);
        MEMBER(InputTextParameters, bool, password, false);
        MEMBER(InputTextParameters, bool, folderButton, false);
        MEMBER(InputTextParameters, std::optional<std::string>, defaultValue, std::nullopt);
        MEMBER(InputTextParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool InputText(InputTextParameters const& parameters, char* buffer, int bufferSize);
    static bool InputText(InputTextParameters const& parameters, std::string& text);

    struct InputFilterParameters
    {
        MEMBER(InputFilterParameters, float, width, 0);
    };
    static bool InputFilter(InputFilterParameters const& parameters, std::string& filter);

    struct InputTextMultilineParameters
    {
        MEMBER(InputTextMultilineParameters, std::string, name, "");
        MEMBER(InputTextMultilineParameters, std::string, hint, "");
        MEMBER(InputTextMultilineParameters, float, textWidth, 100);
        MEMBER(InputTextMultilineParameters, float, height, 100.0f);
    };
    static void InputTextMultiline(InputTextMultilineParameters const& parameters, std::string& text);

    struct ComboParameters
    {
        MEMBER(ComboParameters, std::string, name, "");
        MEMBER(ComboParameters, float, textWidth, 100);
        MEMBER(ComboParameters, bool, readOnly, false);
        MEMBER(ComboParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER(ComboParameters, bool const*, defaultEnabledValue, nullptr);
        MEMBER(ComboParameters, std::vector<std::string>, values, std::vector<std::string>());
        MEMBER(ComboParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Combo(ComboParameters& parameters, int& value, bool* enabled = nullptr);

    struct SwitcherParameters
    {
        MEMBER(SwitcherParameters, std::string, name, "");
        MEMBER(SwitcherParameters, float, width, 0);
        MEMBER(SwitcherParameters, float, textWidth, 100);
        MEMBER(SwitcherParameters, bool, readOnly, false);
        MEMBER(SwitcherParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER(SwitcherParameters, std::vector<std::string>, values, std::vector<std::string>());
        MEMBER(SwitcherParameters, std::optional<int>, disabledValue, std::nullopt);
        MEMBER(SwitcherParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Switcher(SwitcherParameters& parameters, int& value, bool* enabled = nullptr);

    struct ComboColorParameters
    {
        MEMBER(ComboColorParameters, std::string, name, "");
        MEMBER(ComboColorParameters, float, width, 0);
        MEMBER(ComboColorParameters, float, textWidth, 100);
        MEMBER(ComboColorParameters, std::optional<int>, defaultValue, std::nullopt);
        MEMBER(ComboColorParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool ComboColor(ComboColorParameters const& parameters, int& value, bool* enabled = nullptr);
    static bool ComboOptionalColor(ComboColorParameters const& parameters, std::optional<int>& value);

    struct InputColorTransitionParameters
    {
        MEMBER(InputColorTransitionParameters, std::string, name, "");
        MEMBER(InputColorTransitionParameters, int, color, 0);
        MEMBER(InputColorTransitionParameters, float, textWidth, 100);
        MEMBER(InputColorTransitionParameters, int, min, 1);
        MEMBER(InputColorTransitionParameters, int, max, 10000000);
        MEMBER(InputColorTransitionParameters, bool, logarithmic, false);
        MEMBER(InputColorTransitionParameters, bool, infinity, false);
        MEMBER(InputColorTransitionParameters, std::optional<int>, defaultTargetColor, std::nullopt);
        MEMBER(InputColorTransitionParameters, std::optional<int>, defaultTransitionAge, std::nullopt);
        MEMBER(InputColorTransitionParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void InputColorTransition(InputColorTransitionParameters const& parameters, int sourceColor, int& targetColor, int& transitionAge);

    struct CheckboxParameters
    {
        MEMBER(CheckboxParameters, std::string, name, "");
        MEMBER(CheckboxParameters, float, textWidth, 100);
        MEMBER(CheckboxParameters, std::optional<bool>, defaultValue, std::nullopt);
        MEMBER(CheckboxParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Checkbox(CheckboxParameters const& parameters, bool& value);

    struct ToggleButtonParameters
    {
        MEMBER(ToggleButtonParameters, std::string, name, "");
        MEMBER(ToggleButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool ToggleButton(ToggleButtonParameters const& parameters, bool& value);

    struct SelectableButtonParameters 
    {
        MEMBER(SelectableButtonParameters, std::string, name, "");
        MEMBER(SelectableButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER(SelectableButtonParameters, float, width, 0);
    };
    static bool SelectableButton(SelectableButtonParameters const& parameters, bool& value);

    static void Text(std::string const& text);
    static void BoldText(std::string const& text);
    static void MonospaceText(std::string const& text);

    static void BeginMenuBar();
    static void BeginMenu(std::string const& text, bool& toggled, float focus = true);
    struct MenuItemParameters
    {
        MEMBER(MenuItemParameters, std::string, name, "");
        MEMBER(MenuItemParameters, bool, keyCtrl, false);
        MEMBER(MenuItemParameters, bool, keyAlt, false);
        MEMBER(MenuItemParameters, std::optional<ImGuiKey>, key, std::nullopt);
        MEMBER(MenuItemParameters, bool, disabled, false);
        MEMBER(MenuItemParameters, bool, selected, false);
        MEMBER(MenuItemParameters, bool, closeMenuWhenItemClicked, true);
    };
    static void MenuItem(MenuItemParameters const& parameters, std::function<void()> const& action);
    static void MenuSeparator();
    static void EndMenu();
    static void MenuShutdownButton(std::function<void()> const& action);
    static void EndMenuBar();

    struct ColorButtonWithPickerParameters
    {
        MEMBER(ColorButtonWithPickerParameters, std::string, name, "");
        MEMBER(ColorButtonWithPickerParameters, float, textWidth, 100);
        MEMBER(ColorButtonWithPickerParameters, std::optional<FloatColorRGB>, defaultValue, std::nullopt);
        MEMBER(ColorButtonWithPickerParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static void ColorButtonWithPicker(ColorButtonWithPickerParameters const& parameters, FloatColorRGB& color);

    static void NegativeSpacing();
    static void Separator();

    struct MovableHorizontalSeparatorParameters
    {
        MEMBER(MovableHorizontalSeparatorParameters, bool, additive, true);
    };
    static void MovableHorizontalSeparator(MovableHorizontalSeparatorParameters const& parameters, float& height);

    struct MovableVerticalSeparatorParameters
    {
        MEMBER(MovableVerticalSeparatorParameters, bool, additive, true);
        MEMBER(MovableVerticalSeparatorParameters, float, bottomSpace, 0);
    };
    static void MovableVerticalSeparator(MovableVerticalSeparatorParameters const& parameters, float& width);

    static void Group(std::string const& text, std::optional<std::string> const& tooltip = std::nullopt);

    struct ToolbarButtonParameters
    {
        MEMBER(ToolbarButtonParameters, std::string, text, std::string());
        MEMBER(ToolbarButtonParameters, std::optional<std::string>, secondText, std::nullopt);
        MEMBER(ToolbarButtonParameters, RealVector2D, secondTextOffset, (RealVector2D{25.0f, 25.0f}));
        MEMBER(ToolbarButtonParameters, float, secondTextScale, 0.5f);
        MEMBER(ToolbarButtonParameters, bool, disabled, false);
        MEMBER(ToolbarButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool ToolbarButton(ToolbarButtonParameters const& parameters);
    static bool SelectableToolbarButton(std::string const& text, int& value, int selectionValue, int deselectionValue);

    static void VerticalSeparator(float length = 23.0f);
    static void ToolbarSeparator();
    static bool Button(std::string const& text, float size = 0);
    static bool CollapseButton(bool collapsed);

    enum class TreeNodeRank
    {
        Low,
        Default,
        High
    };
    struct TreeNodeParameters
    {
        MEMBER(TreeNodeParameters, std::string, name, "");
        MEMBER(TreeNodeParameters, TreeNodeRank, rank, TreeNodeRank::Default);
        MEMBER(TreeNodeParameters, bool, defaultOpen, true);
        MEMBER(TreeNodeParameters, bool, visible, true);
        MEMBER(TreeNodeParameters, bool, blinkWhenActivated, false);
    };
    static bool BeginTreeNode(TreeNodeParameters const& parameters);
    static void EndTreeNode();

    static void SetFilterText(std::string const& value);
    static void ResetFilterText();

    struct ButtonParameters
    {
        MEMBER(ButtonParameters, std::string, buttonText, "");
        MEMBER(ButtonParameters, std::string, name, "");
        MEMBER(ButtonParameters, float, textWidth, 100);
        MEMBER(ButtonParameters, bool, showDisabledRevertButton, false);
        MEMBER(ButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool Button(ButtonParameters const& parameters);

    struct ActionButtonParameters
    {
        MEMBER(ActionButtonParameters, std::string, buttonText, "");
        MEMBER(ActionButtonParameters, bool, frame, false);
        MEMBER(ActionButtonParameters, std::optional<std::string>, tooltip, std::nullopt);
    };
    static bool ActionButton(ActionButtonParameters const& parameters);

    struct SpinnerParameters
    {
        MEMBER(SpinnerParameters, RealVector2D, pos, RealVector2D());
    };
    static void Spinner(SpinnerParameters const& parameters);

    static void StatusBar(std::vector<std::string> const& textItems);

    static void Tooltip(std::string const& text, bool delay = true, ImGuiHoveredFlags flags = ImGuiHoveredFlags_AllowWhenDisabled);
    static void Tooltip(std::function<std::string()> const& textFunc, bool delay = true);

    static void ConvertRGBtoHSV(uint32_t rgb, float& h, float& s, float& v);

    static void BeginIndent();
    static void EndIndent();

    static bool ShowPreviewDescription(PreviewDescription const& desc, float& zoom, std::optional<int>& selectedNode);

    struct NeuronSelectionParameters
    {
        MEMBER(NeuronSelectionParameters, std::string, name, "");
        MEMBER(NeuronSelectionParameters, float, step, 0.05f);
        MEMBER(NeuronSelectionParameters, std::string, format, "%.2f");
        MEMBER(NeuronSelectionParameters, float, rightMargin, 0);
    };
    static void NeuronSelection(
        NeuronSelectionParameters const& parameters,
        std::vector<float>& weights,
        std::vector<float>& biases,
        std::vector<ActivationFunction>& activationFunctions);

    static void DisabledField();

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

    static void PaddingLeft();

private:

    template <typename Parameter, typename T>
    static bool BasicSlider(Parameter const& parameters, T* value, bool* enabled, bool* pinned);


    template<typename T>
    struct BasicInputColorMatrixParameters
    {
        MEMBER(BasicInputColorMatrixParameters, std::string, name, "");
        MEMBER(BasicInputColorMatrixParameters, T, min, static_cast<T>(0));
        MEMBER(BasicInputColorMatrixParameters, T, max, static_cast<T>(0));
        MEMBER(BasicInputColorMatrixParameters, bool, logarithmic, false);
        MEMBER(BasicInputColorMatrixParameters, std::string, format, "%.2f");
        MEMBER(BasicInputColorMatrixParameters, float, textWidth, 100);
        MEMBER(BasicInputColorMatrixParameters, std::optional<std::vector<std::vector<T>>>, defaultValue, std::nullopt);
        MEMBER(BasicInputColorMatrixParameters, std::optional<std::string>, tooltip, std::nullopt);
        MEMBER(BasicInputColorMatrixParameters, std::optional<std::vector<std::vector<T>>>, disabledValue, std::nullopt);
    };
    template <typename T>
    static void BasicInputColorMatrix(BasicInputColorMatrixParameters<T> const& parameters, T (&value)[MAX_COLORS][MAX_COLORS], bool* enabled = nullptr);

    //static bool BeginTable(std::string const& id, int column, ImGuiTableFlags flags = 0, RealVector2D size = RealVector2D(0.0f, 0.0f));
    //static void EndTable();

    static ImVec2 RotationCenter(ImDrawList* drawList);

    static bool RevertButton(std::string const& id);

private:
    static bool isFilterActive();
    static bool matchWithFilter(std::string const& text);

    static std::vector<FilterStackElement> _filterStack;
    static std::vector<TreeNodeStackElement> _treeNodeStack;
    static std::unordered_map<unsigned int, TreeNodeInfo> _treeNodeInfoById;

    static std::unordered_set<unsigned int> _basicSilderExpanded;

    static int _rotationStartIndex;

    static std::unordered_map<unsigned int, int> _neuronSelectedInput;
    static std::unordered_map<unsigned int, int> _neuronSelectedOutput;

    static bool _menuBarVisible;
    static bool* _menuButtonToggled;
    static bool _menuButtonToToggle;
};
