#include "AlienGui.h"

#include <chrono>
//#include <mdspan>

#include <boost/algorithm/string.hpp>
#include <imgui.h>
#include <imgui_internal.h>

#include <Fonts/IconsFontAwesome5.h>

#include "Base/Math.h"
#include "Base/StringHelper.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/NumberGenerator.h"
#include "EngineInterface/SimulationParameters.h"

#include "GenericFileDialog.h"
#include "StyleRepository.h"
#include "HelpStrings.h"
#include "OverlayController.h"
#include "LayerColorPalette.h"

namespace
{
    auto constexpr HoveredTimer = 0.5f;
}

std::vector<FilterStackElement> AlienGui::_filterStack;
std::unordered_set<unsigned int> AlienGui::_basicSilderExpanded;
std::vector<TreeNodeStackElement> AlienGui::_treeNodeStack;
std::unordered_map<unsigned int, TreeNodeInfo> AlienGui::_treeNodeInfoById;
std::unordered_map<unsigned int, int> AlienGui::_neuronSelectedInput;
std::unordered_map<unsigned int, int> AlienGui::_neuronSelectedOutput;
int AlienGui::_rotationStartIndex = 0;
bool* AlienGui::_menuButtonToggled = nullptr;
bool AlienGui::_menuButtonToToggle = false;
bool AlienGui::_menuBarVisible = false;

void AlienGui::HelpMarker(std::string const& text)
{
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextInfoColor);
    ImGui::Text(ICON_FA_QUESTION_CIRCLE);
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

bool AlienGui::SliderFloat(SliderFloatParameters const& parameters, float* value, bool* enabled, bool* pinned)
{
    return BasicSlider(parameters, value, enabled, pinned);
}

bool AlienGui::SliderInt(SliderIntParameters const& parameters, int* value, bool* enabled, bool* pinned)
{
    return BasicSlider(parameters, value, enabled, pinned);
}

bool AlienGui::SliderFloat2(SliderFloat2Parameters const& parameters, float& valueX, float& valueY)
{
    if (!matchWithFilter(parameters._name)) {
        return false;
    }

    ImGui::PushID(parameters._name.c_str());

    auto constexpr MousePickerButtonWidth = 22.0f;

    auto mousePickerButtonTotalWidth = parameters._getMousePickerEnabledFunc ? scale(MousePickerButtonWidth) + ImGui::GetStyle().FramePadding.x : 0.0f;
    auto sliderWidth = (ImGui::GetContentRegionAvail().x - scale(parameters._textWidth) - mousePickerButtonTotalWidth);
    ImGui::SetNextItemWidth(sliderWidth);
    bool result = ImGui::SliderFloat("##sliderX", &valueX, parameters._min.x, parameters._max.x, parameters._format.c_str(), 0);

    //mouse picker
    if (parameters._getMousePickerEnabledFunc) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - ImGui::GetStyle().FramePadding.x);
        auto mousePickerEnabled = parameters._getMousePickerEnabledFunc.value()();
        if (AlienGui::SelectableButton(AlienGui::SelectableButtonParameters().name(ICON_FA_CROSSHAIRS).width(MousePickerButtonWidth), mousePickerEnabled)) {
            parameters._setMousePickerEnabledFunc.value()(mousePickerEnabled);
            if (mousePickerEnabled) {
                OverlayController::get().showMessage("Select a position in the simulation view");
            }
        }
        AlienGui::Tooltip("Select a position with the mouse");
        if (parameters._getMousePickerEnabledFunc.value()()) {
            if (auto pos = parameters._getMousePickerPositionFunc.value()()) {
                valueX = pos->x;
                valueY = pos->y;
            }
        }
    }

    //revert button
    if (parameters._defaultValue) {
        ImGui::SameLine();

        ImGui::BeginDisabled(valueX == parameters._defaultValue->x && valueY == parameters._defaultValue->y);
        if (AlienGui::RevertButton(parameters._name)) {
            valueX = parameters._defaultValue->x;
            valueY = parameters._defaultValue->y;
        }
        ImGui::EndDisabled();
    }

    //text
    if (!parameters._name.empty()) {
        ImGui::SameLine();
        AlienGui::Text(parameters._name.c_str());
    }

    //tooltip
    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }
    ImGui::SetNextItemWidth(sliderWidth);
    result |= ImGui::SliderFloat("##sliderY", &valueY, parameters._min.y, parameters._max.y, parameters._format.c_str(), 0);

    ImGui::PopID();
    return result;
}

void AlienGui::SliderInputFloat(SliderInputFloatParameters const& parameters, float& value)
{
    auto textWidth = StyleRepository::get().scale(parameters._textWidth);
    auto inputWidth = StyleRepository::get().scale(parameters._inputWidth);

    ImGui::SetNextItemWidth(
        ImGui::GetContentRegionAvail().x - textWidth - inputWidth
        - ImGui::GetStyle().FramePadding.x * 2);
    ImGui::SliderFloat(
        ("##slider" + parameters._name).c_str(), &value, parameters._min, parameters._max, parameters._format.c_str());
    ImGui::SameLine();
    ImGui::SetNextItemWidth(inputWidth);
    ImGui::InputFloat(("##input" + parameters._name).c_str(), &value, 0, 0, parameters._format.c_str());
    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());
}

bool AlienGui::InputInt(InputIntParameters const& parameters, int& value, bool* enabled)
{
    auto textWidth = scale(parameters._textWidth);
    auto infinityButtonWidth = 30;
    auto isInfinity = value == std::numeric_limits<int>::max();
    auto showInfinity = parameters._infinity && (!parameters._readOnly || isInfinity);

    auto result = false;
    if (enabled) {
        result |= ImGui::Checkbox(("##checkbox" + parameters._name).c_str(), enabled);
        if (!(*enabled) && parameters._disabledValue) {
            value = *parameters._disabledValue;
        }
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    auto inputWidth = ImGui::GetContentRegionAvail().x - textWidth;
    if (showInfinity) {
        inputWidth -= scale(infinityButtonWidth) + ImGui::GetStyle().FramePadding.x;
    }

    if (parameters._readOnly) {
        ImGui::BeginDisabled();
    }

    if (!isInfinity) {
        ImGui::SetNextItemWidth(inputWidth);
        result |= ImGui::InputInt(("##" + parameters._name).c_str(), &value, 1, 100, ImGuiInputTextFlags_None);
    } else {
        std::string text = "infinity";
        result |= InputText(InputTextParameters().readOnly(true).width(inputWidth).textWidth(0), text);
    }
    if (parameters._readOnly) {
        ImGui::EndDisabled();
    }
    if (parameters._defaultValue) {
        ImGui::SameLine();
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (RevertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }
    if (showInfinity) {
        ImGui::SameLine();
        MoveTickLeft();

        ImGui::BeginDisabled(parameters._readOnly);
        if (SelectableButton(SelectableButtonParameters().name(ICON_FA_INFINITY).tooltip(parameters._tooltip).width(infinityButtonWidth), isInfinity)) {
            if (isInfinity) {
                value = std::numeric_limits<int>::max();
            } else {
                value = 1;
            }
        }
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    if (enabled) {
        ImGui::EndDisabled();
    }
    AlienGui::Text(parameters._name);
    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }
    return result;
}

namespace
{
    template <typename Parameters, typename T, typename Callable>
    bool optionalWidgetAdaptor(Parameters parameters, std::optional<T>& optionalValue, T const& defaultValue, Callable const& func)
    {
        auto newParameters = parameters;

        auto enabled = optionalValue.has_value();
        auto value = optionalValue.value_or(parameters._defaultValue.value_or(defaultValue));
        auto result = func(newParameters, value, &enabled);
        result |= (optionalValue.has_value() != enabled);
        optionalValue = enabled ? std::make_optional(value) : std::nullopt;
        return result;
    }
}

bool AlienGui::InputOptionalInt(InputIntParameters const& parameters, std::optional<int>& optValue)
{
    return optionalWidgetAdaptor(parameters, optValue, 0, &AlienGui::InputInt);
}

bool AlienGui::InputFloat(InputFloatParameters const& parameters, float& value)
{
    auto textWidth = StyleRepository::get().scale(parameters._textWidth);

    ImGuiInputTextFlags flags = parameters._readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    ImGui::BeginDisabled(parameters._readOnly);
    auto result = ImGui::InputFloat(("##" + parameters._name).c_str(), &value, parameters._step, 0, parameters._format.c_str(), flags);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (RevertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());

    if (parameters._tooltip) {
        HelpMarker(*parameters._tooltip);
    }
    return result;
}

void AlienGui::InputFloat2(InputFloat2Parameters const& parameters, float& value1, float& value2)
{
    if (!matchWithFilter(parameters._name)) {
        return;
    }

    auto textWidth = StyleRepository::get().scale(parameters._textWidth);

    ImGuiInputTextFlags flags = parameters._readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    static float value[2];
    value[0] = value1;
    value[1] = value2;

    ImGui::BeginDisabled(parameters._readOnly);
    ImGui::InputFloat2(("##" + parameters._name).c_str(), value, parameters._format.c_str(), flags);
    ImGui::EndDisabled();

    value1 = value[0];
    value2 = value[1];
    ImGui::SameLine();
    if (parameters._defaultValue1 && parameters._defaultValue2) {
        ImGui::BeginDisabled(value1 == *parameters._defaultValue1 && value2 == *parameters._defaultValue2);
        if (RevertButton(parameters._name)) {
            value1 = *parameters._defaultValue1;
            value2 = *parameters._defaultValue2;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());

    if (parameters._tooltip) {
        HelpMarker(*parameters._tooltip);
    }
}

bool AlienGui::ColorField(uint32_t cellColor, float width, float height)
{
    if (width == 0) {
        width = 30.0f;
    }

    if (height == 0) {
        width = ImGui::GetTextLineHeight() + ImGui::GetStyle().FramePadding.y * 2;
    }

    float h, s, v;
    AlienGui::ConvertRGBtoHSV(cellColor, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, s, v));
    auto result = ImGui::Button("##button", ImVec2(scale(width), scale(height)));
    ImGui::PopStyleColor(3);

    return result;
}

void AlienGui::CheckboxColorMatrix(CheckboxColorMatrixParameters const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS])
{
    BasicInputColorMatrixParameters<bool> basicParameters;
    basicParameters._name = parameters._name;
    basicParameters._textWidth = parameters._textWidth;
    basicParameters._defaultValue = parameters._defaultValue;
    basicParameters._tooltip = parameters._tooltip;
    BasicInputColorMatrix<bool>(basicParameters, value);
}

void AlienGui::InputIntColorMatrix(InputIntColorMatrixParameters const& parameters, int (&value)[MAX_COLORS][MAX_COLORS])
{
    BasicInputColorMatrixParameters<int> basicParameters;
    basicParameters._name = parameters._name;
    basicParameters._min = parameters._min;
    basicParameters._max = parameters._max;
    basicParameters._format = "%d";
    basicParameters._logarithmic = parameters._logarithmic;
    basicParameters._textWidth = parameters._textWidth;
    basicParameters._defaultValue = parameters._defaultValue;
    basicParameters._tooltip = parameters._tooltip;
    BasicInputColorMatrix<int>(basicParameters, value);
}

void AlienGui::InputFloatColorMatrix(InputFloatColorMatrixParameters const& parameters, float (&value)[MAX_COLORS][MAX_COLORS], bool* enabled)
{
    BasicInputColorMatrixParameters<float> basicParameters;
    basicParameters._name = parameters._name; 
    basicParameters._min = parameters._min;
    basicParameters._max = parameters._max;
    basicParameters._logarithmic = parameters._logarithmic;
    basicParameters._format = parameters._format;
    basicParameters._textWidth = parameters._textWidth;
    basicParameters._defaultValue = parameters._defaultValue;
    basicParameters._tooltip = parameters._tooltip;
    basicParameters._disabledValue = parameters._disabledValue;
    BasicInputColorMatrix<float>(basicParameters, value, enabled);
}

bool AlienGui::InputText(InputTextParameters const& parameters, char* buffer, int bufferSize)
{
    if (!matchWithFilter(parameters._name)) {
        return false;
    }

    auto width = parameters._width != 0.0f ? scale(parameters._width) : ImGui::GetContentRegionAvail().x;
    auto folderButtonWidth = parameters._folderButton ? scale(30.0f) + ImGui::GetStyle().FramePadding.x : 0;
    ImGui::SetNextItemWidth(width - scale(parameters._textWidth) - folderButtonWidth);
    if (parameters._monospaceFont) {
        ImGui::PushFont(StyleRepository::get().getMonospaceMediumFont());
    }
    if (parameters._bold) {
        ImGui::PushFont(StyleRepository::get().getSmallBoldFont());
    }
    ImGuiInputTextFlags flags = 0;
    if (parameters._readOnly) {
        flags |= ImGuiInputTextFlags_ReadOnly;
        ImGui::BeginDisabled();
    }
    if (parameters._password) {
        flags |= ImGuiInputTextFlags_Password;
    }
    auto result = [&] {
        if(!parameters._hint.empty()) {
            return ImGui::InputTextWithHint(("##" + parameters._hint).c_str(), parameters._hint.c_str(), buffer, bufferSize, flags);
        }
        return ImGui::InputText(("##" + parameters._name).c_str(), buffer, bufferSize, flags);
    }();
    if (parameters._readOnly) {
        ImGui::EndDisabled();
    }

    if (parameters._monospaceFont || parameters._bold) {
        ImGui::PopFont();
    }
    if (parameters._folderButton) {
        ImGui::SameLine();
        static std::string selectedFolder;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() - ImGui::GetStyle().FramePadding.x);
        if (ImGui::Button(ICON_FA_FOLDER_OPEN, {scale(30.0f), 0})) {
            GenericFileDialog::get().showOpenFileDialog(
                "Select directory", "", std::string(buffer), [&](std::filesystem::path const& path) {
                    selectedFolder = path.string();
                });
        }
        if (!selectedFolder.empty()) {
            auto folderLength = std::min(toInt(selectedFolder.size()), bufferSize);
            selectedFolder.copy(buffer, folderLength);
            if (folderLength < bufferSize) {
                buffer[folderLength] = '\0';
            }
            selectedFolder.clear();
            result = true;
        }
    }
    if (parameters._defaultValue) {
        ImGui::SameLine();
        ImGui::BeginDisabled(std::string(buffer) == *parameters._defaultValue);
        if (RevertButton(parameters._name)) {
            StringHelper::copy(buffer, bufferSize, *parameters._defaultValue);
            result = true;
        }
        ImGui::EndDisabled();
    }
    if (!parameters._name.empty()) {
        ImGui::SameLine();
        AlienGui::Text(parameters._name.c_str());
    }
    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }

    return result;
}

bool AlienGui::InputText(InputTextParameters const& parameters, std::string& text)
{
    static char buffer[1024];
    StringHelper::copy(buffer, IM_ARRAYSIZE(buffer), text);
    auto result = AlienGui::InputText(parameters, buffer, IM_ARRAYSIZE(buffer));
    text = std::string(buffer);

    return result;
}

bool AlienGui::InputFilter(InputFilterParameters const& parameters, std::string& filter)
{
    auto result = AlienGui::InputText(
        AlienGui::InputTextParameters().hint("Filter (case insensitive)").bold(!filter.empty()).textWidth(0).width(parameters._width - 28.0f), filter);
    ImGui::SameLine();

    ImGui::BeginDisabled(filter.empty());
    if (AlienGui::Button(ICON_FA_TIMES)) {
        filter.clear();
        result = true;
    }
    ImGui::EndDisabled();

    return result;
}

void AlienGui::InputTextMultiline(InputTextMultilineParameters const& parameters, std::string& text)
{
    static char buffer[1024*16];
    StringHelper::copy(buffer, IM_ARRAYSIZE(buffer), text);

    auto textWidth = StyleRepository::get().scale(parameters._textWidth);
    auto height = parameters._height == 0
        ? ImGui::GetContentRegionAvail().y
        : StyleRepository::get().scale(parameters._height);

    ImGui::InputTextEx(
        ("##" + parameters._name).c_str(),
        parameters._hint.c_str(),
        buffer,
        IM_ARRAYSIZE(buffer),
        {ImGui::GetContentRegionAvail().x - textWidth, height},
        ImGuiInputTextFlags_Multiline);
    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());

    text = std::string(buffer);
}

namespace
{
    auto vectorGetter = [](void* vec, int idx, const char** outText) {
        auto& vector = *static_cast<std::vector<std::string>*>(vec);
        if (idx < 0 || idx >= static_cast<int>(vector.size())) {
            return false;
        }
        *outText = vector.at(idx).c_str();
        return true;
    };

}

bool AlienGui::Combo(ComboParameters& parameters, int& value, bool* enabled)
{
    if (!matchWithFilter(parameters._name)) {
        return false;
    }

    auto textWidth = StyleRepository::get().scale(parameters._textWidth);

    const char** items = new const char*[parameters._values.size()];
    for (int i = 0; i < parameters._values.size(); ++i) {
        items[i] = parameters._values[i].c_str();
    }

    if (parameters._readOnly) {
        ImGui::BeginDisabled();
    }

    auto result = false;
    if (enabled) {
        result |= ImGui::Checkbox(("##checkbox" + parameters._name).c_str(), enabled);
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    result |= ImGui::Combo(
        ("##" + parameters._name).c_str(),
        &value,
        vectorGetter,
        static_cast<void*>(&parameters._values),
        parameters._values.size());
    ImGui::PopItemWidth();

    if (enabled) {
        ImGui::EndDisabled();
    }

    if (parameters._readOnly) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    if (parameters._defaultValue) {
        auto equalEnabledValue = !parameters._defaultEnabledValue || *parameters._defaultEnabledValue == *enabled;
        ImGui::BeginDisabled(value == *parameters._defaultValue && equalEnabledValue);
        if (RevertButton(parameters._name)) {
            value = *parameters._defaultValue;
            *enabled = *parameters._defaultEnabledValue;
            result = true;
        }
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());

    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }

    delete[] items;
    return result;
}

bool AlienGui::Switcher(SwitcherParameters& parameters, int& value, bool* enabled /*= nullptr*/)
{
    if (!matchWithFilter(parameters._name)) {
        return false;
    }

    ImGui::PushID(parameters._name.c_str());

    if (parameters._readOnly) {
        ImGui::BeginDisabled();
    }

    //enable button
    if (enabled) {
        ImGui::Checkbox("##checkbox", enabled);
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    static auto constexpr buttonWidth = 22.0f;
    auto width = parameters._width != 0.0f ? scale(parameters._width) : ImGui::GetContentRegionAvail().x;
    auto textAndButtonWidth = scale(parameters._textWidth + buttonWidth * 2) + ImGui::GetStyle().FramePadding.x * 2;
    auto switcherWidth = width - textAndButtonWidth;

    auto result = false;
    auto numValues = toInt(parameters._values.size());

    std::string text = parameters._values[value];

    static char buffer[1024];
    StringHelper::copy(buffer, IM_ARRAYSIZE(buffer), text);

    ImGui::SetNextItemWidth(switcherWidth);
    ImGui::InputText(("##" + parameters._name).c_str(), buffer, IM_ARRAYSIZE(buffer), ImGuiInputTextFlags_ReadOnly);

    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - ImGui::GetStyle().FramePadding.x);
    if (ImGui::Button(ICON_FA_CARET_LEFT, ImVec2(scale(buttonWidth), 0))) {
        value = (value + numValues - 1) % numValues;
        result = true;
    }

    ImGui::SameLine();
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - ImGui::GetStyle().FramePadding.x);
    if (ImGui::Button(ICON_FA_CARET_RIGHT, ImVec2(scale(buttonWidth), 0))) {
        value = (value + 1) % numValues;
        result = true;
    }

    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (RevertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }

    if (enabled) {
        ImGui::EndDisabled();
    }
    if (parameters._readOnly) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    AlienGui::Text(parameters._name);

    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }

    ImGui::PopID();
    return result;
}

bool AlienGui::ComboColor(ComboColorParameters const& parameters, int& value, bool* enabled)
{
    if (enabled) {
        ImGui::Checkbox(("##checkbox" + parameters._name).c_str(), enabled);
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    auto width = parameters._width != 0.0f ? scale(parameters._width) : ImGui::GetContentRegionAvail().x;
    auto textWidth = scale(parameters._textWidth);
    auto comboWidth = width - textWidth;
    auto colorFieldWidth1 = comboWidth - scale(25.0f);
    auto colorFieldWidth2 = comboWidth - scale(30.0f);

    const char* items[] = { "##1", "##2", "##3", "##4", "##5", "##6", "##7" };

    ImVec2 comboPos = ImGui::GetCursorScreenPos();

    ImGui::SetNextItemWidth(comboWidth);
    if (ImGui::BeginCombo(("##" + parameters._name).c_str(), "")) {
        for (int n = 0; n < MAX_COLORS; ++n) {
            bool isSelected = (value == n);

            if (ImGui::Selectable(items[n], isSelected)) {
                value = n;
            }
            ImGui::SameLine();
            ColorField(Const::IndividualCellColors[n], colorFieldWidth1, ImGui::GetTextLineHeight());
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();

    ImGuiStyle& style = ImGui::GetStyle();
    float h, s, v;
    AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[value], h, s, v);
    if (enabled && !(*enabled)) {
        s = 0;
        v = 0.2f;
    }
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(comboPos.x + style.FramePadding.x, comboPos.y + style.FramePadding.y),
        ImVec2(comboPos.x + style.FramePadding.x + colorFieldWidth2, comboPos.y + style.FramePadding.y + ImGui::GetTextLineHeight()),
        ImColor::HSV(h, s, v));


    if (enabled) {
        ImGui::EndDisabled();
    }
    AlienGui::Text(parameters._name);

    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }

    return true;
}

bool AlienGui::ComboOptionalColor(ComboColorParameters const& parameters, std::optional<int>& value)
{
    return optionalWidgetAdaptor(parameters, value, 0, &AlienGui::ComboColor);
}

void AlienGui::InputColorTransition(InputColorTransitionParameters const& parameters, int sourceColor, int& targetColor, int& transitionAge)
{
    if (!matchWithFilter(parameters._name)) {
        return;
    }

    //source color field
    ImGui::PushID(sourceColor);
    AlienGui::ColorField(Const::IndividualCellColors[sourceColor]);
    ImGui::SameLine();

    //combo for target color
    AlienGui::Text(ICON_FA_LONG_ARROW_ALT_RIGHT);
    ImGui::SameLine();
    ImGui::PushID("color");
    AlienGui::ComboColor(AlienGui::ComboColorParameters().width(70.0f).textWidth(0), targetColor);
    ImGui::PopID();


    //slider for transition age
    ImGui::PushID(2);

    ImGui::SameLine();
    auto width = scale(parameters._textWidth);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - width);
    std::string format = "%d";
    if (parameters._infinity && transitionAge == Infinity<int>::value) {
        format = "infinity";
        transitionAge = parameters._max;
    }
    ImGui::SliderInt(
        ("##" + parameters._name).c_str(),
        &transitionAge,
        parameters._min,
        parameters._max,
        format.c_str(),
        parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
    if (parameters._infinity && transitionAge == parameters._max) {
        format = "infinity";
        transitionAge = Infinity<int>::value;
    }
    if (parameters._defaultTransitionAge && parameters._defaultTargetColor) {
        ImGui::SameLine();
        ImGui::BeginDisabled(transitionAge == *parameters._defaultTransitionAge && targetColor == *parameters._defaultTargetColor);
        if (RevertButton(parameters._name)) {
            transitionAge = *parameters._defaultTransitionAge;
            targetColor = *parameters._defaultTargetColor;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());

    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }
    ImGui::PopID();
    ImGui::PopID();
}

bool AlienGui::Checkbox(CheckboxParameters const& parameters, bool& value)
{
    if (!matchWithFilter(parameters._name)) {
        return false;
    }
    ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x - scale(parameters._textWidth) - scale(30.0f), 0.0f));
    ImGui::SameLine();

    auto result = ImGui::Checkbox(("##" + parameters._name).c_str(), &value);

    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (RevertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }

    return result;
}

bool AlienGui::SelectableButton(SelectableButtonParameters const& parameters, bool& value)
{
    auto buttonColor = ImColor(ImGui::GetStyle().Colors[ImGuiCol_Button]);
    auto buttonColorHovered = ImColor(ImGui::GetStyle().Colors[ImGuiCol_ButtonHovered]);
    auto buttonColorActive = ImColor(ImGui::GetStyle().Colors[ImGuiCol_ButtonActive]);
    if (value) {
        buttonColor = buttonColorActive;
        buttonColorHovered = buttonColorActive;
    }

    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)buttonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)buttonColorHovered);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)buttonColorActive);
    auto result = ImGui::Button(parameters._name.c_str(), {scale(parameters._width), 0});
    if (result) {
        value = !value;
    }
    ImGui::PopStyleColor(3);

    return result;
}

void AlienGui::Text(std::string const& text)
{
    auto refPos = ImGui::GetCursorScreenPos();
    ImGui::TextUnformatted(text.c_str());
    if (isFilterActive()) {
        auto [beforeMatch, match] = StringHelper::decomposeCaseInsensitiveMatch(text, _filterStack.back().text);
        if (!match.empty()) {
            auto prefixSize = ImGui::CalcTextSize(beforeMatch.c_str());
            ImGui::GetWindowDrawList()->AddText(
                ImGui::GetFont(),
                ImGui::GetFontSize(),
                {refPos.x + prefixSize.x + 1, refPos.y + ImGui::GetStyle().FramePadding.y},
                ImGui::GetColorU32(ImGuiCol_Text),
                match.c_str());
            ImGui::GetWindowDrawList()->AddText(
                ImGui::GetFont(),
                ImGui::GetFontSize(),
                {refPos.x + prefixSize.x, refPos.y + ImGui::GetStyle().FramePadding.y + 1},
                ImGui::GetColorU32(ImGuiCol_Text),
                match.c_str());
        }
    }
}

void AlienGui::BoldText(std::string const& text)
{
    ImGui::PushFont(StyleRepository::get().getSmallBoldFont());
    AlienGui::Text(text);
    ImGui::PopFont();
}

void AlienGui::MonospaceText(std::string const& text)
{
    ImGui::PushFont(StyleRepository::get().getMonospaceMediumFont());
    ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::MonospaceColor);
    Text(text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

void AlienGui::BeginMenuBar()
{
    _menuBarVisible = ImGui::BeginMainMenuBar();
}

void AlienGui::BeginMenu(std::string const& text, bool& toggled, float focus)
{
    if (!_menuBarVisible) {
        return ;
    }
    _menuButtonToggled = &toggled;
    _menuButtonToToggle = false;

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2);
    const auto active = toggled;
    if (active) {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::MenuButtonActiveColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::MenuButtonHoveredColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::MenuButtonHoveredColor);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::MenuButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::MenuButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::MenuButtonColor);
    }

    auto pos = ImGui::GetCursorPos();
    if (AlienGui::Button(text.c_str())) {
        toggled = !toggled;
    }
    if (ImGui::IsItemHovered()) {
        toggled = true;
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);

    if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        toggled = false;
    }

    if (toggled) {
        bool open = true;
        ImVec2 buttonPos{pos.x, pos.y};
        ImVec2 buttonSize = ImGui::GetItemRectSize();

        auto height = ImGui::GetWindowHeight();
        ImVec2 windowPos{pos.x, pos.y + height};
        ImGui::SetNextWindowPos(windowPos);
        if (focus) {
            ImGui::SetNextWindowFocus();
        }
        if (ImGui::Begin((text + "##1").c_str(), &open, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize)) {

            auto mousePos = ImGui::GetMousePos();
            auto windowSize = ImGui::GetWindowSize();
            if (/*ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||*/
                ((mousePos.x < windowPos.x || mousePos.y < windowPos.y || mousePos.x > windowPos.x + windowSize.x
                    || mousePos.y > windowPos.y + windowSize.y)
                && (mousePos.x < buttonPos.x || mousePos.y < buttonPos.y || mousePos.x > buttonPos.x + buttonSize.x
                    || mousePos.y > buttonPos.y + buttonSize.y))) {
                EndMenu();
                toggled = false;
            }
        } else {
            toggled = false;
        }
    }
}

void AlienGui::MenuItem(MenuItemParameters const& parameters, std::function<void()> const& action)
{
    if (_menuBarVisible) {
        std::vector<std::string> keyStringParts;
        if (parameters._keyCtrl) {
            keyStringParts.emplace_back("CTRL");
        }
        if (parameters._keyAlt) {
            keyStringParts.emplace_back("ALT");
        }
        if (*_menuButtonToggled) {
            if (parameters._disabled) {
                ImGui::BeginDisabled();
            }
            if (parameters._key.has_value()) {
                if (*parameters._key >= ImGuiKey_0 && *parameters._key <= ImGuiKey_Z) {
                    static std::vector<std::string> const keyMap = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
                                                                    "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};
                    keyStringParts.emplace_back(keyMap.at(*parameters._key - ImGuiKey_0));
                }
                if (*parameters._key == ImGuiKey_Space) {
                    keyStringParts.emplace_back("Space");
                }
                if (*parameters._key == ImGuiKey_Escape) {
                    keyStringParts.emplace_back("Esc");
                }
                if (*parameters._key == ImGuiKey_Delete) {
                    keyStringParts.emplace_back("Delete");
                }
            }
            auto keyString = boost::join(keyStringParts, "+");
            if (ImGui::MenuItem(parameters._name.c_str(), keyString.c_str(), parameters._selected)) {
                action();
                if (parameters._closeMenuWhenItemClicked) {
                    _menuButtonToToggle = true;
                }
            }
            if (parameters._disabled) {
                ImGui::EndDisabled();
            }
        }
    }
    if (!parameters._keyAlt && !parameters._keyCtrl && parameters._key == ImGuiKey_0) {
        return;
    }
    auto const& io = ImGui::GetIO();
    if (parameters._key.has_value() && !parameters._disabled && !io.WantCaptureKeyboard && io.KeyCtrl == parameters._keyCtrl && io.KeyAlt == parameters._keyAlt
        && ImGui::IsKeyPressed(*parameters._key)) {
        action();
    }
}

void AlienGui::MenuSeparator()
{
    if (_menuBarVisible && * _menuButtonToggled) {
        ImGui::Separator();
    }
}

void AlienGui::EndMenu()
{
    if (!_menuBarVisible) {
        return;
    }
    if (*_menuButtonToggled) {
        ImGui::End();
    }
    if (_menuButtonToToggle) {
        *_menuButtonToggled = !*_menuButtonToggled;
    }
}

void AlienGui::MenuShutdownButton(std::function<void()> const& action)
{
    if (!_menuBarVisible) {
        return;
    }
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ImportantButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ImportantButtonHoveredColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ImportantButtonActiveColor);
    if (ImGui::Button(ICON_FA_POWER_OFF)) {
        action();
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
}

void AlienGui::EndMenuBar()
{
    if (_menuBarVisible) {
        ImGui::EndMainMenuBar();
    }
}

void AlienGui::ColorButtonWithPicker(ColorButtonWithPickerParameters const& parameters, FloatColorRGB& color)
{
    if (!matchWithFilter(parameters._name)) {
        return;
    }

    static FloatColorRGB backupColor;
    static LayerColorPalette layerColorPalette;

    auto savedPalette = layerColorPalette.getReference();

    ImVec4 imGuiColor = ImColor(color.r, color.g, color.b);
    ImVec4 imGuiBackupColor = ImColor(backupColor.r, backupColor.g, backupColor.b);
    ImVec4 imGuiSavedPalette[32];
    for (int i = 0; i < IM_ARRAYSIZE(imGuiSavedPalette); ++i) {
        imGuiSavedPalette[i].x = savedPalette[i].r;
        imGuiSavedPalette[i].y = savedPalette[i].g;
        imGuiSavedPalette[i].z = savedPalette[i].b;
        imGuiSavedPalette[i].w = 1.0f;
    }

    bool openColorPicker = ImGui::ColorButton(
        ("##" + parameters._name).c_str(),
        imGuiColor,
        ImGuiColorEditFlags_NoBorder,
        {ImGui::GetContentRegionAvail().x - StyleRepository::get().scale(parameters._textWidth), 0});
    if (openColorPicker) {
        ImGui::OpenPopup("colorpicker");
        imGuiBackupColor = imGuiColor;
    }
    if (ImGui::BeginPopup("colorpicker")) {
        ImGui::Text("Please choose a color");
        ImGui::Separator();
        ImGui::ColorPicker4("##picker", (float*)&imGuiColor, ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview);
        ImGui::SameLine();

        ImGui::BeginGroup();  // Lock X position
        ImGui::Text("Current");
        ImGui::ColorButton("##current", imGuiColor, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40));
        ImGui::Text("Previous");
        if (ImGui::ColorButton("##previous", imGuiBackupColor, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40))) {
            imGuiColor = imGuiBackupColor;
        }
        ImGui::Separator();
        ImGui::Text("Palette");
        for (int n = 0; n < IM_ARRAYSIZE(imGuiSavedPalette); n++) {
            ImGui::PushID(n);
            if ((n % 8) != 0)
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.y);

            ImGuiColorEditFlags paletteButtonFlags = ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip;
            if (ImGui::ColorButton("##palette", imGuiSavedPalette[n], paletteButtonFlags, ImVec2(20, 20)))
                imGuiColor = ImVec4(imGuiSavedPalette[n].x, imGuiSavedPalette[n].y, imGuiSavedPalette[n].z,
                                    imGuiColor.w);  // Preserve alpha!

            // Allow user to drop colors into each palette entry. Note that ColorButton() is already a
            // drag source by default, unless specifying the ImGuiColorEditFlags_NoDragDrop flag.
            if (ImGui::BeginDragDropTarget()) {
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F))
                    memcpy((float*)&imGuiSavedPalette[n], payload->Data, sizeof(float) * 3);
                if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_4F))
                    memcpy((float*)&imGuiSavedPalette[n], payload->Data, sizeof(float) * 4);
                ImGui::EndDragDropTarget();
            }

            ImGui::PopID();
        }
        ImGui::EndGroup();
        ImGui::EndPopup();
    }
    color.r = ImColor(imGuiColor).Value.x;
    color.g = ImColor(imGuiColor).Value.y;
    color.b = ImColor(imGuiColor).Value.z;
    backupColor.r = ImColor(imGuiBackupColor).Value.x;
    backupColor.g = ImColor(imGuiBackupColor).Value.y;
    backupColor.b = ImColor(imGuiBackupColor).Value.z;
    for (int i = 0; i < IM_ARRAYSIZE(imGuiSavedPalette); ++i) {
        savedPalette[i].r = ImColor(imGuiSavedPalette[i]).Value.x;
        savedPalette[i].g = ImColor(imGuiSavedPalette[i]).Value.y;
        savedPalette[i].b = ImColor(imGuiSavedPalette[i]).Value.z;
    }

    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(color == *parameters._defaultValue);
        if (RevertButton(parameters._name)) {
            color = *parameters._defaultValue;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }
}

void AlienGui::MoveTickLeft()
{
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - ImGui::GetStyle().FramePadding.x);
}

void AlienGui::MoveTickUp()
{
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetStyle().FramePadding.y);
}

void AlienGui::Separator()
{
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();
}

void AlienGui::MovableHorizontalSeparator(MovableHorizontalSeparatorParameters const& parameters, float& height)
{
    ImGui::Button("###MovableHorizontalSeparator", ImVec2(-1, scale(5.0f)));
    if (ImGui::IsItemActive()) {
        if (parameters._additive) {
            height += ImGui::GetIO().MouseDelta.y;
        } else {
            height -= ImGui::GetIO().MouseDelta.y;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
    }
}

void AlienGui::MovableVerticalSeparator(MovableVerticalSeparatorParameters const& parameters, float& width)
{
    auto sizeAvailable = ImGui::GetContentRegionAvail();
    ImGui::Button("###MovableVerticalSeparator", ImVec2(scale(5.0f), sizeAvailable.y - scale(parameters._bottomSpace)));
    if (ImGui::IsItemActive()) {
        if (parameters._additive) {
            width += ImGui::GetIO().MouseDelta.x;
        } else {
            width -= ImGui::GetIO().MouseDelta.x;
        }
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
}

void AlienGui::Group(std::string const& text, std::optional<std::string> const& tooltip)
{
    auto drawList = ImGui::GetWindowDrawList();
    auto style = ImGui::GetStyle();

    ImGui::Spacing();
    ImGui::Spacing();

    auto cursorPos = ImGui::GetCursorScreenPos();
    drawList->AddRectFilled(
        ImVec2(cursorPos.x, cursorPos.y - style.FramePadding.y),
        ImVec2(cursorPos.x + scale(ImGui::GetContentRegionAvail().x), cursorPos.y + ImGui::GetTextLineHeight() + style.FramePadding.y),
        Const::GroupColor,
        2.0f);
    ImGui::TextUnformatted((" "  + text).c_str());
    if (tooltip.has_value()) {
        AlienGui::HelpMarker(*tooltip);
    }

    ImGui::Spacing();
    ImGui::Spacing();
}

bool AlienGui::ToolbarButton(ToolbarButtonParameters const& parameters)
{
    auto id = std::to_string(ImGui::GetID(parameters._text.c_str()));
    if (parameters._secondText.has_value()) {
        id += parameters._secondText.value();
    }
    ImGui::PushID(id.c_str());

    ImGui::PushFont(StyleRepository::get().getIconFont());
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, {0.5f, 0.75f});
    auto color = Const::ToolbarButtonTextColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);

    ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(Const::ToolbarButtonBackgroundColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>( ImColor::HSV(h, s, v * 0.3f)));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, static_cast<ImVec4> (ImColor::HSV(h, s, v * 0.45f)));

    ImGui::PushStyleColor(ImGuiCol_Text, static_cast<ImVec4> (Const::ToolbarButtonTextColor));
    auto buttonSize = scale(40.0f);

    ImGui::BeginDisabled(parameters._disabled);

    auto pos = ImGui::GetCursorScreenPos();
    auto result = ImGui::Button(parameters._text.c_str(), {buttonSize, buttonSize});

    if (parameters._secondText.has_value()) {
        ImGui::GetWindowDrawList()->AddText(
            ImGui::GetFont(),
            ImGui::GetFontSize() * parameters._secondTextScale,
            {pos.x + scale(parameters._secondTextOffset.x), pos.y + scale(parameters._secondTextOffset.y)},
            ImGui::GetColorU32(ImGuiCol_Text),
            parameters._secondText->c_str());
    }
    ImGui::EndDisabled();

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar();
    ImGui::PopFont();

    if (parameters._tooltip) {
        AlienGui::Tooltip(*parameters._tooltip);
    }
    ImGui::PopID();
    return result;
}

bool AlienGui::SelectableToolbarButton(std::string const& text, int& value, int selectionValue, int deselectionValue)
{
    auto id = std::to_string(ImGui::GetID(text.c_str()));

    ImGui::PushFont(StyleRepository::get().getIconFont());
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, {0.5f, 0.75f});
    auto color = Const::ToolbarButtonTextColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);

    auto buttonColor = Const::ToolbarButtonBackgroundColor;
    auto buttonColorHovered = ImColor::HSV(h, s, v * 0.3f);
    auto buttonColorActive = ImColor::HSV(h, s, v * 0.45f);
    if (value == selectionValue) {
        buttonColor = buttonColorActive;
        buttonColorHovered = buttonColorActive;
    }

    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)buttonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)buttonColorHovered);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)buttonColorActive);

    ImGui::PushStyleColor(ImGuiCol_Text, static_cast<ImVec4>(Const::ToolbarButtonTextColor));
    auto buttonSize = scale(40.0f);
    auto result = ImGui::Button(text.c_str(), {buttonSize, buttonSize});
    if (result) {
        if (value == selectionValue) {
            value = deselectionValue;
        } else {
            value = selectionValue;
        }
    }

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar();
    ImGui::PopFont();
    return result;
}

void AlienGui::VerticalSeparator(float length)
{
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    auto cursorPos = ImGui::GetCursorScreenPos();
    auto color = ImColor(ImGui::GetStyle().Colors[ImGuiCol_Border]);
    color.Value.w *= ImGui::GetStyle().Alpha;
    drawList->AddLine(ImVec2(cursorPos.x, cursorPos.y - ImGui::GetStyle().FramePadding.y), ImVec2(cursorPos.x, cursorPos.y + scale(length)), color, 2.0f);
    ImGui::Dummy(ImVec2(ImGui::GetStyle().FramePadding.x * 2, 1));
}

void AlienGui::ToolbarSeparator()
{
    VerticalSeparator(40.0f);
}

bool AlienGui::Button(std::string const& text, float size)
{
/*
    auto color = Const::ButtonColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.4f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 0.6f));
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::ButtonColor);
*/
    auto result = ImGui::Button(text.c_str(), ImVec2(scale(size), 0));
    /*
    ImGui::PopStyleColor(4);
*/
    return result;
}

bool AlienGui::CollapseButton(bool collapsed)
{
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0, 0, 0));
    auto result = ImGui::ArrowButton("##collapseButton", collapsed ? ImGuiDir_Right : ImGuiDir_Down);
    ImGui::PopStyleColor(1);
    return result;
}

bool AlienGui::BeginTreeNode(TreeNodeParameters const& parameters)
{
    CHECK(_treeNodeStack.size() < 10);

    auto forceTreeNodeOpen = isFilterActive();

    if (matchWithFilter(parameters._name)) {
        auto newFilterElement = !_filterStack.empty() ? _filterStack.back() : FilterStackElement();
        newFilterElement.alreadyMatched = true;
        _filterStack.emplace_back(newFilterElement);
    } else if (!_filterStack.empty()) {
        _filterStack.emplace_back(_filterStack.back());
    } else {
        // insert dummy element since AlienImGui:EndTreeNode pops last element
        _filterStack.emplace_back(FilterStackElement());
    }

    auto id = ImGui::GetID(parameters._name.c_str());

    if (!parameters._visible) {
        _treeNodeStack.emplace_back(0.0f, id, false);
        _treeNodeInfoById.insert_or_assign(id, TreeNodeInfo{.invisibleTimepoint = std::chrono::steady_clock::now(), .isEmpty = false});
        return false;
    }

    TreeNodeInfo info = _treeNodeInfoById.contains(id) ? _treeNodeInfoById.at(id) : TreeNodeInfo();
    int highlightCountdown = 0;
    if (parameters._blinkWhenActivated) {
        highlightCountdown = std::max(0, toInt(1000 - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - info.invisibleTimepoint)
                      .count()));
        if (highlightCountdown > 0) {
            ImGui::SetScrollHereY();
        }
    }

    if (isFilterActive() && info.isEmpty) {
        _treeNodeStack.emplace_back(ImGui::GetCursorPosY(), id, false);
        return true;
    }

    ImColor header, headerHovered, headerActive;
    if (parameters._rank == TreeNodeRank::High) {
        header = Const::TreeNodeHighColor;
        headerHovered = Const::TreeNodeHighHoveredColor;
        headerActive = Const::TreeNodeHighActiveColor;
    } else if (parameters._rank == TreeNodeRank::Default) {
        header = Const::TreeNodeDefaultColor;
        headerHovered = Const::TreeNodeDefaultHoveredColor;
        headerActive = Const::TreeNodeDefaultActiveColor;
    } else if (parameters._rank == TreeNodeRank::Low) {
        header = Const::TreeNodeLowColor;
        headerHovered = Const::TreeNodeLowHoveredColor;
        headerActive = Const::TreeNodeLowActiveColor;
    }
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(header.Value.x, header.Value.y, header.Value.z, h, s, v);
    v = std::min(1.0f, v * (1.0f + toFloat(highlightCountdown) / 1000));
    h = h + toFloat(highlightCountdown) / 5000;
    header = ImColor::HSV(h, s, v);

    ImGui::PushStyleColor(ImGuiCol_Header, header.Value);
    ImGui::PushStyleColor(ImGuiCol_HeaderHovered, headerHovered.Value);
    ImGui::PushStyleColor(ImGuiCol_HeaderActive, headerActive.Value);
    ImGuiTreeNodeFlags treeNodeClosedFlags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_Framed;
    ImGuiTreeNodeFlags treeNodeOpenFlags = treeNodeClosedFlags | ImGuiTreeNodeFlags_DefaultOpen;

    if (forceTreeNodeOpen) {
        ImGui::SetNextItemOpen(true);
        treeNodeClosedFlags |= ImGuiTreeNodeFlags_Bullet;
        treeNodeOpenFlags |= ImGuiTreeNodeFlags_Bullet;
    }
    ImGui::PushFont(StyleRepository::get().getSmallBoldFont());
    bool result = ImGui::TreeNodeEx(parameters._name.c_str(), parameters._defaultOpen ? treeNodeOpenFlags : treeNodeClosedFlags);
    ImGui::PopFont();
    ImGui::PopStyleColor(3);

    _treeNodeStack.emplace_back(ImGui::GetCursorPosY(), id, result);
    return result;
}

void AlienGui::EndTreeNode()
{
    _filterStack.pop_back();
    TreeNodeStackElement stackElement = _treeNodeStack.back();
    _treeNodeStack.pop_back();

    _treeNodeInfoById[stackElement.treeNodeId].isEmpty =
        stackElement.treeNodeStartCursorPosY != 0 && stackElement.treeNodeStartCursorPosY == ImGui::GetCursorPosY();

    if (stackElement.isOpen) {
        ImGui::TreePop();
    }
}

void AlienGui::SetFilterText(std::string const& value)
{
    CHECK(_filterStack.size() < 10);

    _filterStack.emplace_back(value);
}

void AlienGui::ResetFilterText()
{
    _filterStack.pop_back();
}

bool AlienGui::Button(ButtonParameters const& parameters)
{
    auto width = ImGui::GetContentRegionAvail().x - StyleRepository::get().scale(parameters._textWidth);
    auto result = ImGui::Button(parameters._buttonText.c_str(), {width, 0});
    ImGui::SameLine();

    if (parameters._showDisabledRevertButton) {
        ImGui::BeginDisabled(true);
        RevertButton(parameters._name);
        ImGui::EndDisabled();
        ImGui::SameLine();
    }
    AlienGui::Text(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }
    return result;
}

bool AlienGui::ActionButton(ActionButtonParameters const& parameters)
{
    ImGui::PushStyleColor(ImGuiCol_Text, Const::ActionButtonTextColor.Value);
    ImGui::PushStyleColor(ImGuiCol_Button, Const::ActionButtonBackgroundColor.Value);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Const::ActionButtonHoveredColor.Value);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, Const::ActionButtonActiveColor.Value);
    auto size = ImGui::CalcTextSize(parameters._buttonText.c_str());
    auto padding = ImGui::GetStyle().FramePadding;
    size.x += padding.x * 2;
    size.y += padding.y * 2;
    auto cursorPos = ImGui::GetCursorScreenPos();
    auto result = ImGui::Button(parameters._buttonText.c_str(), size);
    if (parameters._frame) {
        ImGui::GetWindowDrawList()->AddRect(cursorPos, {cursorPos.x + size.x, cursorPos.y + size.y}, ImColor::HSV(0.54f, 0.43f, 0.5f));
    }
    ImGui::PopStyleColor(4);

    if (parameters._tooltip) {
        AlienGui::Tooltip(*parameters._tooltip);
    }

    return result;
}

void AlienGui::Spinner(SpinnerParameters const& parameters)
{
    static std::optional<std::chrono::steady_clock::time_point> spinnerRefTimepoint;
    static float spinnerAngle = 0;
    if (!spinnerRefTimepoint.has_value()) {
        spinnerRefTimepoint = std::chrono::steady_clock::now();
    }
    auto now = std::chrono::steady_clock::now();
    auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - *spinnerRefTimepoint).count());

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    AlienGui::RotateStart(drawList);
    auto font = StyleRepository::get().getIconFont();
    auto text = ICON_FA_SPINNER;
    ImVec4 clipRect(-100000.0f, -100000.0f, 100000.0f, 100000.0f);
    font->RenderText(
        drawList,
        scale(30.0f),
        {parameters._pos.x - scale(15.0f), parameters._pos.y - scale(80.0f)},
        ImColor::HSV(0.5f, 0.1f, 1.0f, std::min(1.0f, duration / 500)),
        clipRect,
        text,
        text + strlen(text),
        0.0f,
        false);

    auto angle = sinf(duration * Const::DegToRad / 10 - Const::Pi / 2) * 4 + 6.0f;
    spinnerAngle += angle;
    AlienGui::RotateEnd(spinnerAngle, drawList);
}

void AlienGui::StatusBar(std::vector<std::string> const& textItems)
{
    std::string text;
    for (auto const& textItem : textItems) {
        text += " " ICON_FA_INFO_CIRCLE " " + textItem;
    }
    AlienGui::Separator();
    ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::MonospaceColor);
    //ImGui::PushStyleColor(ImGuiCol_Text, Const::StatusBarTextColor.Value);
    AlienGui::Text(text);
    ImGui::PopStyleColor();
}

void AlienGui::Tooltip(std::string const& text, bool delay, ImGuiHoveredFlags flags)
{
    if (ImGui::IsItemHovered(flags) && (!delay || (delay && GImGui->HoveredIdTimer > HoveredTimer))) {
        ImGui::BeginTooltip();
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TooltipTextColor.Value);
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        ImGui::EndTooltip();
    }
}

void AlienGui::Tooltip(std::function<std::string()> const& textFunc, bool delay)
{
    if (ImGui::IsItemHovered() && (!delay || (delay && GImGui->HoveredIdTimer > HoveredTimer))) {
        Tooltip(textFunc(), delay);
    }
}

void AlienGui::ConvertRGBtoHSV(uint32_t rgb, float& h, float& s, float& v)
{
    return ImGui::ColorConvertRGBtoHSV(
        toFloat((rgb >> 16) & 0xff) / 255, toFloat((rgb >> 8) & 0xff) / 255, toFloat((rgb & 0xff)) / 255, h, s, v);
}

void AlienGui::BeginIndent()
{
    ImGui::Dummy(ImVec2(scale(22), 0));
    ImGui::SameLine();
    ImGui::BeginGroup();
}

void AlienGui::EndIndent()
{
    ImGui::EndGroup();
}

bool AlienGui::ToggleButton(ToggleButtonParameters const& parameters, bool& value)
{
    auto origValue = value;
    ImVec4* colors = ImGui::GetStyle().Colors;
    ImVec2 p = ImGui::GetCursorScreenPos();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    float height = ImGui::GetFrameHeight();
    float width = height * 1.55f;
    float radius = height * 0.50f*0.8f;
    height = height * 0.8f;

    ImGui::InvisibleButton(parameters._name.c_str(), ImVec2(width, height));
    if (ImGui::IsItemClicked()) {
        value = !value;
    }

    auto color = Const::ToggleColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);

    if (ImGui::IsItemHovered()) {
        drawList->AddRectFilled(
            p,
            ImVec2(p.x + width, p.y + height),
            ImGui::GetColorU32(value ? (ImU32)ImColor::HSV(h, s * 0.9f, v * 0.8f) : (ImU32)ImColor::HSV(h, s * 0.9f, v * 0.4f)),
            height * 0.5f);
    } else {
        drawList->AddRectFilled(
            p,
            ImVec2(p.x + width, p.y + height),
            ImGui::GetColorU32(value ? (ImU32)ImColor::HSV(h, s * 0.6f, v * 0.7f) : (ImU32)ImColor::HSV(h, s * 0.6f, v * 0.3f)),
            height * 0.50f);
    }
    drawList->AddCircleFilled(ImVec2(p.x + radius + (value ? 1 : 0) * (width - radius * 2.0f), p.y + radius), radius - 1.5f, IM_COL32(20, 20, 20, 255));
    drawList->AddCircleFilled(ImVec2(p.x + radius + (value ? 1 : 0) * (width - radius * 2.0f), p.y + radius), radius - 2.5f, IM_COL32(255, 255, 255, 255));

    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }

    return value != origValue;
}

bool AlienGui::ShowPreviewDescription(PreviewDescription const& desc, float& zoom, std::optional<int>& selectedNode)
{
    auto constexpr ZoomLevelForLabels = 16.0f;
    auto constexpr ZoomLevelForConnections = 8.0f;
    auto const LineThickness = scale(2.0f);

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    auto const cellSize = scale(zoom);

    auto drawTextWithShadow = [&drawList, &cellSize](std::string const& text, float posX, float posY) {
        drawList->AddText(
            StyleRepository::get().getLargeFont(), cellSize / 2, {posX + 1.0f, posY + 1.0f}, Const::ExecutionNumberOverlayShadowColor, text.c_str());
        drawList->AddText(StyleRepository::get().getLargeFont(), cellSize / 2, {posX, posY}, Const::ExecutionNumberOverlayColor, text.c_str());
    };

    auto result = false;

    auto color = ImGui::GetStyle().Colors[ImGuiCol_WindowBg];
    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();

    RealVector2D upperLeft;
    RealVector2D lowerRight;
    for (auto const& cell : desc.cells) {
        if (cell.pos.x < upperLeft.x) {
            upperLeft.x = cell.pos.x;
        }
        if (cell.pos.y < upperLeft.y) {
            upperLeft.y = cell.pos.y;
        }
        if (cell.pos.x > lowerRight.x) {
            lowerRight.x = cell.pos.x;
        }
        if (cell.pos.y > lowerRight.y) {
            lowerRight.y = cell.pos.y;
        }
    }
    RealVector2D previewSize = (lowerRight - upperLeft) * cellSize + RealVector2D(cellSize, cellSize) * 2;

    auto mousePos = ImGui::GetMousePos();
    auto clickedOnPreviewWindow = ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mousePos.x >= windowPos.x && mousePos.y >= windowPos.y
        && mousePos.x <= windowPos.x + windowSize.x && mousePos.y <= windowPos.y + windowSize.y;
    ImGui::SetCursorPos({std::max(0.0f, windowSize.x - previewSize.x) / 2, std::max(0.0f, windowSize.y - previewSize.y) / 2});
    if (ImGui::BeginChild("##genome", ImVec2(previewSize.x, previewSize.y), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        auto windowPos = ImGui::GetWindowPos();
        RealVector2D offset{windowPos.x + cellSize, windowPos.y + cellSize};

        ImGui::SetCursorPos({previewSize.x - 1, previewSize.y - 1});

        //draw cells
        for (auto const& cell : desc.cells) {
            auto cellPos = (cell.pos - upperLeft) * cellSize + offset;
            float h, s, v;
            AlienGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell.color], h, s, v);

            auto cellRadiusFactor = zoom > ZoomLevelForConnections ? 0.25f : 0.5f;
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * cellRadiusFactor, ImColor::HSV(h, s * 1.2f, v * 1.0f));

            if (selectedNode && cell.nodeIndex == *selectedNode) {
                if (zoom > ZoomLevelForLabels) {
                    drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor(1.0f, 1.0f, 1.0f));
                } else {
                    drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor::HSV(h, s * 0.8f, v * 1.2f));
                }
            }

            if (clickedOnPreviewWindow) {
                if (mousePos.x >= cellPos.x - cellSize / 2 && mousePos.y >= cellPos.y - cellSize / 2 && mousePos.x <= cellPos.x + cellSize / 2
                    && mousePos.y <= cellPos.y + cellSize / 2) {
                    selectedNode = cell.nodeIndex;
                    result = true;
                }
            }

        }

        //draw symbols
        for (auto const& symbol : desc.symbols) {
            auto pos = (symbol.pos - upperLeft) * cellSize + offset;
            switch (symbol.type) {
            case SymbolPreviewDescription::Type::Dot: {
                auto cellRadiusFactor = zoom > ZoomLevelForConnections ? 0.15f : 0.35f;
                drawList->AddCircleFilled({pos.x, pos.y}, cellSize * cellRadiusFactor, Const::GenomePreviewDotSymbolColor);
            } break;
            case SymbolPreviewDescription::Type::Infinity: {
                if (zoom > ZoomLevelForConnections) {
                    drawList->AddText(
                        StyleRepository::get().getIconFont(),
                        cellSize / 2,
                        {pos.x - cellSize * 0.4f, pos.y - cellSize * 0.2f},
                        Const::GenomePreviewInfinitySymbolColor,
                        ICON_FA_INFINITY);
                }
            } break;
            }
        }

        //draw cell connections
        if (zoom > ZoomLevelForConnections) {
            for (auto const& connection : desc.connections) {
                auto cellPos1 = (connection.cell1 - upperLeft) * cellSize + offset;
                auto cellPos2 = (connection.cell2 - upperLeft) * cellSize + offset;

                auto direction = cellPos1 - cellPos2;

                Math::normalize(direction);
                auto connectionStartPos = cellPos1 - direction * cellSize / 4;
                auto connectionEndPos = cellPos2 + direction * cellSize / 4;
                drawList->AddLine(
                    {connectionStartPos.x, connectionStartPos.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

                if (connection.arrowToCell1) {
                    auto arrowPartDirection1 = RealVector2D{-direction.x + direction.y, -direction.x - direction.y};
                    auto arrowPartStart1 = connectionStartPos + arrowPartDirection1 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart1.x, arrowPartStart1.y},
                        {connectionStartPos.x, connectionStartPos.y},
                        Const::GenomePreviewConnectionColor,
                        LineThickness);

                    auto arrowPartDirection2 = RealVector2D{-direction.x - direction.y, direction.x - direction.y};
                    auto arrowPartStart2 = connectionStartPos + arrowPartDirection2 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart2.x, arrowPartStart2.y},
                        {connectionStartPos.x, connectionStartPos.y},
                        Const::GenomePreviewConnectionColor,
                        LineThickness);
                }

                if (connection.arrowToCell2) {
                    auto arrowPartDirection1 = RealVector2D{direction.x - direction.y, direction.x + direction.y};
                    auto arrowPartStart1 = connectionEndPos + arrowPartDirection1 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart1.x, arrowPartStart1.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);

                    auto arrowPartDirection2 = RealVector2D{direction.x + direction.y, -direction.x + direction.y};
                    auto arrowPartStart2 = connectionEndPos + arrowPartDirection2 * cellSize / 8;
                    drawList->AddLine(
                        {arrowPartStart2.x, arrowPartStart2.y}, {connectionEndPos.x, connectionEndPos.y}, Const::GenomePreviewConnectionColor, LineThickness);
                }
            }
        }

        //draw cell infos (start/end marks and multiple constructor marks)
        if (zoom > ZoomLevelForLabels) {
            for (auto const& cell : desc.cells) {
                auto cellPos = (cell.pos - upperLeft) * cellSize + offset;
                auto length = cellSize / 4;
                if (cell.partStart !=  cell.partEnd) {
                    drawList->AddTriangleFilled(
                        {cellPos.x + length, cellPos.y},
                        {cellPos.x + length * 2, cellPos.y - length / 2},
                        {cellPos.x + length * 2, cellPos.y + length / 2},
                        cell.partStart ? Const::GenomePreviewStartColor : Const::GenomePreviewEndColor);
                }
                if (cell.partStart && cell.partEnd) {
                    drawList->AddTriangleFilled(
                        {cellPos.x + length, cellPos.y - length},
                        {cellPos.x + length * 2, cellPos.y - length * 3  / 2},
                        {cellPos.x + length * 2, cellPos.y - length / 2},
                        Const::GenomePreviewStartColor);
                    drawList->AddTriangleFilled(
                        {cellPos.x + length, cellPos.y + length},
                        {cellPos.x + length * 2, cellPos.y + length / 2},
                        {cellPos.x + length * 2, cellPos.y + length * 3 / 2},
                        Const::GenomePreviewEndColor);
                }
                if (cell.multipleConstructor) {
                    drawList->AddLine(
                        {cellPos.x + length, cellPos.y + length},
                        {cellPos.x + length * 2, cellPos.y + length},
                        Const::GenomePreviewMultipleConstructorColor,
                        LineThickness);
                    drawList->AddLine(
                        {cellPos.x + length * 1.5f, cellPos.y + length / 2},
                        {cellPos.x + length * 1.5f, cellPos.y + length * 1.5f},
                        Const::GenomePreviewMultipleConstructorColor,
                        LineThickness);
                }
                if (cell.selfReplicator) {
                    drawList->AddText(
                        StyleRepository::get().getIconFont(),
                        cellSize / 4,
                        {cellPos.x - length * 2, cellPos.y + length},
                        Const::GenomePreviewSelfReplicatorColor,
                        ICON_FA_CLONE);
                }
            }
        }

    }
    ImGui::EndChild();

    //zoom buttons
    ImGui::SetCursorPos({ImGui::GetScrollX() + scale(10), ImGui::GetScrollY() + windowSize.y - scale(40)});
    if (ImGui::BeginChild("##buttons", ImVec2(scale(100), scale(30)), false)) {
        ImGui::SetCursorPos({0, 0});
        ImGui::PushStyleColor(ImGuiCol_Button, color);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, color);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, color);
        ImGui::PushID(1);
        if (ImGui::Button(ICON_FA_SEARCH_PLUS)) {
            zoom *= 1.5f;
        }
        ImGui::PopID();
        ImGui::SameLine();
        ImGui::PushID(2);
        if (ImGui::Button(ICON_FA_SEARCH_MINUS)) {
            zoom /= 1.5f;
        }
        ImGui::PopID();
        ImGui::PopStyleColor(3);
    }
    ImGui::EndChild();

    return result;
}

namespace
{
    template<typename T>
    int& getIdBasedValue(std::unordered_map<unsigned int, T>& idToValueMap, T const& defaultValue)
    {
        auto id = ImGui::GetID("");
        if (!idToValueMap.contains(id)) {
            idToValueMap[id] = defaultValue;
        }
        return idToValueMap.at(id);
    }

}

void AlienGui::NeuronSelection(
    NeuronSelectionParameters const& parameters,
    std::vector<float>& weights,
    std::vector<float>& biases,
    std::vector<ActivationFunction>& activationFunctions)
{
    // #TODO GCC incompatibily:
    // auto weights_span = std::mdspan(weights.data(), MAX_CHANNELS, MAX_CHANNELS);
    auto& selectedInput = getIdBasedValue(_neuronSelectedInput, 0);
    auto& selectedOutput = getIdBasedValue(_neuronSelectedOutput, 0);
    auto setDefaultColors = [] {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ToggleButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ToggleButtonHoveredColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ToggleButtonHoveredColor);
    };
    auto setHightlightingColors = [] {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ToggleButtonActiveColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ToggleButtonActiveColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ToggleButtonActiveColor);
    };
    RealVector2D const ioButtonSize{scale(90.0f), scale(26.0f)};
    RealVector2D const plotSize{scale(50.0f), scale(23.0f)};
    auto const rightMargin = scale(parameters._rightMargin);
    auto const biasFieldWidth = ImGui::GetStyle().FramePadding.x * 2;

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    auto windowPos = ImGui::GetWindowPos();

    RealVector2D inputPos[MAX_CHANNELS];
    RealVector2D outputPos[MAX_CHANNELS];

    //draw buttons and save positions to visualize weights
    for (int i = 0; i < MAX_CHANNELS; ++i) {

        auto buttonStartPos = ImGui::GetCursorPos();

        //input button
        i == selectedInput ? setHightlightingColors() : setDefaultColors();
        if (ImGui::Button(("Input #" + std::to_string(i)).c_str(), {ioButtonSize.x, ioButtonSize.y})) {
            selectedInput = i;
        }
        ImGui::PopStyleColor(3);

        Tooltip(Const::NeuronInputTooltipByChannel[i], false);

        inputPos[i] = RealVector2D(
            windowPos.x - ImGui::GetScrollX() + buttonStartPos.x + ioButtonSize.x, windowPos.y - ImGui::GetScrollY() + buttonStartPos.y + ioButtonSize.y / 2);

        ImGui::SameLine(0, ImGui::GetContentRegionAvail().x - ioButtonSize.x * 2 - plotSize.x - ImGui::GetStyle().FramePadding.x - rightMargin);
        buttonStartPos = ImGui::GetCursorPos();
        outputPos[i] = RealVector2D(
            windowPos.x - ImGui::GetScrollX() + buttonStartPos.x - biasFieldWidth - ImGui::GetStyle().FramePadding.x,
            windowPos.y - ImGui::GetScrollY() + buttonStartPos.y + ioButtonSize.y / 2);

        //output button
        i == selectedOutput ? setHightlightingColors() : setDefaultColors();
        if (ImGui::Button(("Output #" + std::to_string(i)).c_str(), {ioButtonSize.x, ioButtonSize.y})) {
            selectedOutput = i;
        }
        ImGui::PopStyleColor(3);
        Tooltip(Const::NeuronOutputTooltipByChannel[i], false);
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        for (int j = 0; j < MAX_CHANNELS; ++j) {
            // #TODO GCC incompatibily:
            // if (std::abs(weights_span[j, i]) > NEAR_ZERO) {
            if (std::abs(weights[j * MAX_CHANNELS + i]) <= NEAR_ZERO) {
                continue;
            }
            drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, Const::NeuronEditorConnectionColor, 2.0f);
        }
    }
    auto calcColor = [](float value) {
        auto factor = std::min(1.0f, std::abs(value));
        if (value > NEAR_ZERO) {
            return ImColor::HSV(0.61f, 0.5f, 0.8f * factor);
        } else if (value < -NEAR_ZERO) {
            return ImColor::HSV(0.0f, 0.5f, 0.8f * factor);
        } else {
            return ImColor::HSV(0.0f, 0.0f, 0.1f);
        }
    };

    //visualize weights
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        for (int j = 0; j < MAX_CHANNELS; ++j) {
            // #TODO GCC incompatibily:
            // if (std::abs(weights_span[j, i]) <= NEAR_ZERO) {
            if (std::abs(weights[j * MAX_CHANNELS + i]) <= NEAR_ZERO) {
                continue;
            }
            // #TODO GCC incompatibily:
            // auto thickness = std::min(4.0f, std::abs(weights_span[j, i]));
            auto thickness = std::min(4.0f, std::abs(weights[j * MAX_CHANNELS + i]));
            // #TODO GCC incompatibily:
            // drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, calcColor(weights_span[j, i]), thickness);
            drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, calcColor(weights[j * MAX_CHANNELS + i]), thickness);
        }
    }

    //visualize activation functions
    auto calcPlotPosition = [&](RealVector2D const& refPos, float x, ActivationFunction activationFunction) {
        float value = 0;
        switch (activationFunction) {
        case ActivationFunction_Sigmoid:
            value = Math::sigmoid(x);
            break;
        case ActivationFunction_BinaryStep:
            value = Math::binaryStep(x);
            break;
        case ActivationFunction_Identity:
            value = x / 4;
            break;
        case ActivationFunction_Abs:
            value = std::abs(x) / 4;
            break;
        case ActivationFunction_Gaussian:
            value = Math::gaussian(x);
            break;
        }
        return RealVector2D{refPos.x + plotSize.x / 2 + x * plotSize.x / 8, refPos.y - value * plotSize.y / 2};
    };
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        std::optional<RealVector2D> lastPos;
        RealVector2D refPos{outputPos[i].x + ioButtonSize.x + biasFieldWidth + ImGui::GetStyle().FramePadding.x * 2, outputPos[i].y};
        for (float dx = 0; dx <= plotSize.x + NEAR_ZERO; dx += plotSize.x / 8) {
            auto color = std::abs(dx - plotSize.x / 2) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
            drawList->AddLine({refPos.x + dx, refPos.y - plotSize.y / 2}, {refPos.x + dx, refPos.y + plotSize.y / 2}, color, 1.0f);
        }
        for (float dy = -plotSize.y / 2; dy <= plotSize.y / 2 + NEAR_ZERO; dy += plotSize.y / 6) {
            auto color = std::abs(dy) < NEAR_ZERO ? Const::NeuronEditorZeroLinePlotColor : Const::NeuronEditorGridColor;
            drawList->AddLine({refPos.x, refPos.y + dy}, {refPos.x + plotSize.x, refPos.y + dy}, color, 1.0f);
        }
        for (float dx = -4.0f; dx < 4.0f; dx += 0.2f) {
            RealVector2D pos = calcPlotPosition(refPos, dx, activationFunctions[i]);
            if (lastPos) {
                drawList->AddLine({lastPos->x, lastPos->y}, {pos.x, pos.y}, Const::NeuronEditorPlotColor, 1.0f);
            }
            lastPos = pos;
        }
    }

    //visualize biases
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        drawList->AddRectFilled(
            {outputPos[i].x, outputPos[i].y - biasFieldWidth}, {outputPos[i].x + biasFieldWidth, outputPos[i].y + biasFieldWidth}, calcColor(biases[i]));
    }

    //draw selection
    drawList->AddRectFilled(
        {outputPos[selectedOutput].x, outputPos[selectedOutput].y - biasFieldWidth},
        {outputPos[selectedOutput].x + biasFieldWidth, outputPos[selectedOutput].y + biasFieldWidth},
        ImColor::HSV(0.0f, 0.0f, 1.0f, 0.35f));
    drawList->AddLine(
        {inputPos[selectedInput].x, inputPos[selectedInput].y}, {outputPos[selectedOutput].x, outputPos[selectedOutput].y}, ImColor::HSV(0.0f, 0.0f, 1.0f, 0.35f), 8.0f);

    auto const editorWidth = ImGui::GetContentRegionAvail().x - rightMargin;
    auto const editorColumnWidth = 280.0f;
    auto const editorColumnTextWidth = 155.0f;
    auto const numWidgets = 3;
    auto numColumns = DynamicTableLayout::calcNumColumns(editorWidth - ImGui::GetStyle().FramePadding.x * 4, editorColumnWidth);
    auto numRows = numWidgets / numColumns;
    if (numWidgets % numColumns != 0) {
        ++numRows;
    }
    if (ImGui::BeginChild("##", ImVec2(editorWidth, scale(toFloat(numRows) * 26.0f + 18.0f + 28.0f)), true)) {
        DynamicTableLayout table(editorColumnWidth);
        if (table.begin()) {
            int activationFunction = activationFunctions.at(selectedOutput);
            AlienGui::Combo(
                AlienGui::ComboParameters()
                    .name("Activation function")
                    .textWidth(editorColumnTextWidth)
                    .values(Const::ActivationFunctionStrings)
                    .tooltip(Const::GenomeNeuronActivationFunctionTooltip),
                activationFunction);
            activationFunctions.at(selectedOutput) = static_cast<ActivationFunction>(activationFunction);
            table.next();
            // #TODO GCC incompatibily:
            // AlienGui::InputFloat(
            //     AlienGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(editorColumnTextWidth).tooltip(Const::GenomeNeuronWeightAndBiasTooltip),
            //     weights_span[selectedOutput, selectedInput]);
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(editorColumnTextWidth).tooltip(Const::GenomeNeuronWeightAndBiasTooltip),
                weights[selectedOutput * MAX_CHANNELS + selectedInput]);
            table.next();
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters().name("Bias").step(0.05f).textWidth(editorColumnTextWidth).tooltip(Const::GenomeNeuronWeightAndBiasTooltip),
                biases.at(selectedOutput));
            table.end();
        }
        if (AlienGui::Button("Clear")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    // #TODO GCC incompatibily:
                    // weights_span[i, j] = 0;
                    weights[i * MAX_CHANNELS + j] = 0;
                }
                biases[i] = 0;
                activationFunctions[i] = ActivationFunction_Sigmoid;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Identity")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    // #TODO GCC incompatibily:
                    // weights_span[i, j] = i == j ? 1.0f : 0.0f;
                    weights[i * MAX_CHANNELS + j] = i == j ? 1.0f : 0.0f;
                }
                biases[i] = 0.0f;
                activationFunctions[i] = ActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienGui::Button("Randomize")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    // #TODO GCC incompatibily:
                    // weights_span[i, j] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                    weights[i * MAX_CHANNELS + j] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                }
                biases[i] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                activationFunctions[i] = NumberGenerator::get().getRandomInt(ActivationFunction_Count);
            }
        }
    }
    ImGui::EndChild();
}


void AlienGui::DisabledField()
{
    auto startPos = ImGui::GetCursorScreenPos();
    auto size = ImGui::GetContentRegionAvail();
    ImGui::GetWindowDrawList()->AddRectFilledMultiColor(
        {startPos.x, startPos.y},
        {startPos.x + size.x, startPos.y + size.y},
        Const::DisabledOverlayColor1,
        Const::DisabledOverlayColor2,
        Const::DisabledOverlayColor1,
        Const::DisabledOverlayColor2);
}

namespace
{
    template <typename T>
    std::string toString(T const& value, std::string const& format, bool allowInfinity = false, bool tryMaintainFormat = false)
    {
        if (allowInfinity && value == Infinity<T>::value) {
            return "infinity";
        }
        if (tryMaintainFormat) {
            return format;
        }
        char result[16];
        snprintf(result, sizeof(result), format.c_str(), value);
        return std::string(result);
    }
}

template <typename Parameter, typename T>
bool AlienGui::BasicSlider(Parameter const& parameters, T* value, bool* enabled, bool* pinned)
{
    if (!matchWithFilter(parameters._name)) {
        return false;
    }

    auto constexpr PinnedButtonWidth = 22.0f;

    ImGui::PushID(parameters._name.c_str());

    if (parameters._readOnly) {
        ImGui::BeginDisabled();
    }

    //enable button
    if (enabled) {
        ImGui::Checkbox("##checkbox", enabled);
        if (!(*enabled)) {
            auto numRows = parameters._colorDependence ? MAX_COLORS : 1;
            for (int row = 0; row < numRows; ++row) {
                value[row] = parameters._disabledValue[row];
            }
        }
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    //color dependent button
    auto toggleButtonId = ImGui::GetID("expanded");
    auto isExpanded = _basicSilderExpanded.contains(toggleButtonId);
    if (parameters._colorDependence) {
        auto buttonResult = Button(isExpanded ? ICON_FA_MINUS_SQUARE "##toggle" : ICON_FA_PLUS_SQUARE "##toggle");
        if (buttonResult) {
            if (isExpanded) {
                _basicSilderExpanded.erase(toggleButtonId);
            } else {
                _basicSilderExpanded.insert(toggleButtonId);
            }
        }
        ImGui::SameLine();
    }

    bool result = false;
    float sliderPosX;
    for (int color = 0; color < MAX_COLORS; ++color) {
        if (color > 0) {
            if (!parameters._colorDependence) {
                break;
            }
            if (parameters._colorDependence && isExpanded == false) {
                break;
            }
        }
        if (color == 0) {
            sliderPosX = ImGui::GetCursorPosX();
        } else {
            ImGui::SetCursorPosX(sliderPosX);
        }

        //color field
        ImGui::PushID(color);
        auto pinnedButtonWidth = pinned ? scale(PinnedButtonWidth) + ImGui::GetStyle().FramePadding.x : 0.0f;
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scale(parameters._textWidth) - pinnedButtonWidth);
        if (parameters._colorDependence && isExpanded) {
            AlienGui::ColorField(Const::IndividualCellColors[color], 0);
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scale(parameters._textWidth));
        }

        //slider
        T sliderValue;
        T minValue = value[0], maxValue = value[0];
        int sliderValueColor = 0;
        std::string format;
        if (parameters._colorDependence && !isExpanded) {
            for (int color = 1; color < MAX_COLORS; ++color) {
                maxValue = std::max(maxValue, value[color]);
                if (minValue > value[color]) {
                    minValue = value[color];
                    sliderValueColor = color;
                }
            }
            if (minValue != maxValue) {
                if constexpr (std::is_same<T, float>()) {
                    format = parameters._format + " ... " + toString(maxValue, parameters._format, parameters._infinity, false);
                } else {
                    format = toString(minValue, parameters._format, parameters._infinity, false) + " ... "
                        + toString(maxValue, parameters._format, parameters._infinity, false);
                }
            } else {
                format = toString(value[color], parameters._format, parameters._infinity, true);
            }
            sliderValue = minValue;
        }
        else {
            format = toString(value[color], parameters._format, parameters._infinity, true);
            sliderValue = value[color];
            sliderValueColor = color;
        }
        if (parameters._infinity && value[color] == Infinity<T>::value) {
            value[color] = parameters._max;
        }

        if constexpr (std::is_same<T, float>()) {
            result |= ImGui::SliderFloat(
                "##slider", &sliderValue, parameters._min, parameters._max, format.c_str(), parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
        }
        if constexpr (std::is_same<T, int>()) {
            result |= ImGui::SliderInt(
                "##slider", &sliderValue, parameters._min, parameters._max, format.c_str(), parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
        }
        value[sliderValueColor] = sliderValue;

        if (parameters._infinity && value[color] == parameters._max) {
            value[color] = Infinity<T>::value;
        }
        if (parameters._colorDependence && !isExpanded && result) {
            for (int color = 1; color < MAX_COLORS; ++color) {
                value[color] = value[0];
            }
        }

        ImGui::PopID();

        if (color == 0) {

            //pin button
            if (pinned) {
                ImGui::SameLine();
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() - ImGui::GetStyle().FramePadding.x);
                AlienGui::SelectableButton(AlienGui::SelectableButtonParameters().name(ICON_FA_THUMBTACK).width(PinnedButtonWidth), *pinned);
            }

            //revert button
            if (parameters._defaultValue) {
                ImGui::SameLine();

                auto equal = true;
                auto numRows = parameters._colorDependence ? MAX_COLORS : 1;
                for (int row = 0; row < numRows; ++row) {
                    if (value[row] != parameters._defaultValue[row]) {
                        equal = false;
                        break;
                    }
                }
                if (parameters._defaultEnabledValue) {
                    if (*parameters._defaultEnabledValue != *enabled) {
                        equal = false;
                    }
                }
                ImGui::BeginDisabled(equal);
                if (RevertButton(parameters._name)) {
                    for (int row = 0; row < numRows; ++row) {
                        value[row] = parameters._defaultValue[row];
                    }
                    if (parameters._defaultEnabledValue) {
                        *enabled = *parameters._defaultEnabledValue;
                    }
                    result = true;
                }
                ImGui::EndDisabled();
            }

            //text
            if (!parameters._name.empty()) {
                ImGui::SameLine();
                if (enabled) {
                    ImGui::EndDisabled();
                }
                if (parameters._readOnly) {
                    ImGui::EndDisabled();
                }
                AlienGui::Text(parameters._name);
                if (parameters._readOnly) {
                    ImGui::BeginDisabled();
                }
                if (enabled) {
                    ImGui::BeginDisabled(!(*enabled));
                }
            }

            //tooltip
            if (parameters._tooltip) {
                if (enabled) {
                    ImGui::EndDisabled();
                }
                AlienGui::HelpMarker(*parameters._tooltip);
                if (enabled) {
                    ImGui::BeginDisabled(!(*enabled));
                }
            }
        }
    }
    if (enabled) {
        ImGui::EndDisabled();
    }
    if (parameters._readOnly) {
        ImGui::EndDisabled();
    }
    ImGui::PopID();
    return result;
}

template <typename T>
void AlienGui::BasicInputColorMatrix(BasicInputColorMatrixParameters<T> const& parameters, T (&value)[MAX_COLORS][MAX_COLORS], bool* enabled)
{
    if (!matchWithFilter(parameters._name)) {
        return;
    }

    ImGui::PushID(parameters._name.c_str());

    if (enabled) {
        ImGui::Checkbox("##cellTypeAttackerGenomeComplexityBonus", enabled);
        if (!(*enabled)) {
            for (int i = 0; i < MAX_COLORS; ++i) {
                for (int j = 0; j < MAX_COLORS; ++j) {
                    value[i][j] = (*parameters._disabledValue)[i][j];
                    value[i][j] = (*parameters._disabledValue)[i][j];
                }
            }
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!(*enabled));
    }

    auto toggleButtonId = ImGui::GetID("expanded");
    auto isExpanded = _basicSilderExpanded.contains(toggleButtonId);
    auto buttonResult = Button(isExpanded ? ICON_FA_MINUS_SQUARE "##toggle" : ICON_FA_PLUS_SQUARE "##toggle");
    if (buttonResult) {
        if (isExpanded) {
            _basicSilderExpanded.erase(toggleButtonId);
        } else {
            _basicSilderExpanded.insert(toggleButtonId);
        }
    }
    auto textWidth = StyleRepository::get().scale(parameters._textWidth);

    ImGui::SameLine();

    if (isExpanded) {
        ImGui::BeginGroup();
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + scale(115.0f));
        ImGui::Text("[target cell color]");

        auto startPos = ImGui::GetCursorPos();

        ImGui::SetCursorPos({startPos.x - scale(48), startPos.y + scale(121)});
        auto drawList = ImGui::GetWindowDrawList();
        RotateStart(drawList);
        ImGui::Text("[cell color]");
        RotateEnd(-90.0f, drawList);

        ImGui::SetCursorPos(startPos);

        //color matrix
        if (ImGui::BeginTable(("##" + parameters._name).c_str(), MAX_COLORS + 1, 0, ImVec2(ImGui::GetContentRegionAvail().x - textWidth, 0))) {
            for (int row = 0; row < MAX_COLORS + 1; ++row) {
                ImGui::PushID(row);
                ImGui::SetCursorPosX(startPos.x);
                for (int col = 0; col < MAX_COLORS + 1; ++col) {
                    ImGui::PushID(col);
                    ImGui::TableNextColumn();
                    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                    if (row == 0 && col > 0) {
                        ColorField(Const::IndividualCellColors[col - 1], -1);
                    } else if (row > 0 && col == 0) {
                        ColorField(Const::IndividualCellColors[row - 1], -1);
                    } else if (row > 0 && col > 0) {
                        if constexpr (std::is_same<T, float>()) {
                            ImGui::InputFloat(("##" + parameters._name).c_str(), &value[row - 1][col - 1], 0, 0, parameters._format.c_str());
                        }
                        if constexpr (std::is_same<T, int>()) {
                            ImGui::InputInt(("##" + parameters._name).c_str(), &value[row - 1][col - 1], 0, 0);
                        }
                        if constexpr (std::is_same<T, bool>()) {
                            ImGui::Checkbox(("##" + parameters._name).c_str(), &value[row - 1][col - 1]);
                        }
                    }
                    ImGui::PopID();
                }
                ImGui::TableNextRow();
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
        ImGui::EndGroup();
    } else {

        //collapsed view
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
        if constexpr (std::is_same<T, bool>()) {
            static bool test = false;
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.5f, 0.5f));
            if (ImGui::Button("Define matrix", ImVec2(ImGui::GetContentRegionAvail().x - textWidth, 0))) {
                _basicSilderExpanded.insert(toggleButtonId);
            }
            ImGui::PopStyleVar();
        } else {
            auto format = parameters._format;
            T sliderValue;
            T minValue = value[0][0], maxValue = value[0][0];
            for (int i = 0; i < MAX_COLORS; ++i) {
                for (int j = 0; j < MAX_COLORS; ++j) {
                    maxValue = std::max(maxValue, value[i][j]);
                    minValue = std::min(minValue, value[i][j]);
                }
            }

            if (minValue != maxValue) {
                if constexpr (std::is_same<T, float>()) {
                    format = parameters._format + " ... " + toString(maxValue, parameters._format, false, false);
                } else {
                    format = std::to_string(minValue) + " ... " + std::to_string(maxValue);
                }
            } else {
                format = toString(value[0][0], parameters._format, false, true);
            }
            sliderValue = minValue;

            auto sliderMoved = false;
            if constexpr (std::is_same<T, float>()) {
                sliderMoved |= ImGui::SliderFloat(
                    "##slider", &sliderValue, parameters._min, parameters._max, format.c_str(), parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
            }
            if constexpr (std::is_same<T, int>()) {
                sliderMoved |= ImGui::SliderInt(
                    "##slider", &sliderValue, parameters._min, parameters._max, format.c_str(), parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
            }
            if (sliderMoved) {
                for (int i = 0; i < MAX_COLORS; ++i) {
                    for (int j = 0; j < MAX_COLORS; ++j) {
                        value[i][j] = sliderValue;
                    }
                }
            }
        }
    }

    ImGui::SameLine();
    if (parameters._defaultValue) {
        bool changed = false;
        for (int row = 0; row < MAX_COLORS; ++row) {
            for (int col = 0; col < MAX_COLORS; ++col) {
                if (value[row][col] != (*parameters._defaultValue)[row][col]) {
                    changed = true;
                }
            }
        }
        ImGui::BeginDisabled(!changed);
        if (RevertButton(parameters._name)) {
            for (int row = 0; row < MAX_COLORS; ++row) {
                for (int col = 0; col < MAX_COLORS; ++col) {
                    value[row][col] = (*parameters._defaultValue)[row][col];
                }
            }
        }
        ImGui::EndDisabled();
    }

    if (enabled) {
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    AlienGui::Text(parameters._name.c_str());

    if (parameters._tooltip) {
        AlienGui::HelpMarker(*parameters._tooltip);
    }

    ImGui::PopID();
}

//RotateStart, RotationCenter, etc. are taken from https://gist.github.com/carasuca/e72aacadcf6cf8139de46f97158f790f
//>>>>>>>>>>
void AlienGui::RotateStart(ImDrawList* drawList)
{
    _rotationStartIndex = drawList->VtxBuffer.Size;
}

ImVec2 AlienGui::RotationCenter(ImDrawList* drawList)
{
    ImVec2 l(FLT_MAX, FLT_MAX), u(-FLT_MAX, -FLT_MAX);  // bounds

    const auto& buf = drawList->VtxBuffer;
    for (int i = _rotationStartIndex; i < buf.Size; i++)
        l = ImMin(l, buf[i].pos), u = ImMax(u, buf[i].pos);

    return ImVec2((l.x + u.x) / 2, (l.y + u.y) / 2);  // or use _ClipRectStack?
}

bool AlienGui::RevertButton(std::string const& id)
{
    auto result = ImGui::Button((ICON_FA_UNDO "##" + id).c_str());
    AlienGui::Tooltip("Revert changes", true, ImGuiHoveredFlags_None);
    return result;
}

bool AlienGui::isFilterActive()
{
    if (_filterStack.empty()) {
        return false;
    }
    return !_filterStack.back().text.empty();
}

bool AlienGui::matchWithFilter(std::string const& text)
{
    if (!isFilterActive()) {
        return true;
    }
    auto filterElement = _filterStack.back();
    if (filterElement.alreadyMatched) {
        return true;
    }
    return StringHelper::containsCaseInsensitive(text, filterElement.text);
}

namespace
{
    ImVec2 operator-(const ImVec2& l, const ImVec2& r)
    {
        return {l.x - r.x, l.y - r.y};
    }
}

void AlienGui::RotateEnd(float angle, ImDrawList* drawList)
{
    auto center = RotationCenter(drawList);
    float s = sin((-angle + 90.0f) * Const::DegToRad), c = cos((-angle + 90.0f) * Const::DegToRad);
    center = ImRotate(center, s, c) - center;

    auto& buf = drawList->VtxBuffer;
    for (int i = _rotationStartIndex; i < buf.Size; i++) {
        buf[i].pos = ImRotate(buf[i].pos, s, c) - center;
    }
}

//<<<<<<<<<<

int AlienGui::DynamicTableLayout::calcNumColumns(float tableWidth, float columnWidth)
{
    return std::max(toInt(tableWidth / scale(columnWidth)), 1);
}

AlienGui::DynamicTableLayout::DynamicTableLayout(float columnWidth)
    : _columnWidth(columnWidth)
{
    _numColumns = calcNumColumns(ImGui::GetContentRegionAvail().x, columnWidth);
}

bool AlienGui::DynamicTableLayout::begin()
{
    auto result = ImGui::BeginTable("##", _numColumns, ImGuiTableFlags_None);
    if (result) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
    }
    return result;
}
void AlienGui::DynamicTableLayout::end()
{
    ImGui::EndTable();
}

void AlienGui::DynamicTableLayout::next()
{
    auto currentCol = (++_elementNumber) % _numColumns;
    if (currentCol > 0) {
        ImGui::TableSetColumnIndex(currentCol);
        AlienGui::VerticalSeparator();
        ImGui::SameLine();
    } else {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
    }
}
