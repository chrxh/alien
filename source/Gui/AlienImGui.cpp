#include "AlienImGui.h"

#include <chrono>

#include <boost/algorithm/string.hpp>
#include <imgui.h>
#include <imgui_internal.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/StringHelper.h"
#include "Base/Math.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/EngineConstants.h"
#include "EngineInterface/SimulationParameters.h"

#include "CellFunctionStrings.h"
#include "StyleRepository.h"
#include "HelpStrings.h"
#include "Base/NumberGenerator.h"

namespace
{
    auto constexpr HoveredTimer = 0.5f;
}

std::unordered_set<unsigned int> AlienImGui::_basicSilderExpanded;
std::unordered_map<unsigned int, int> AlienImGui::_neuronSelectedInput;
std::unordered_map<unsigned int, int> AlienImGui::_neuronSelectedOutput;
int AlienImGui::_rotationStartIndex;


void AlienImGui::HelpMarker(std::string const& text)
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

bool AlienImGui::SliderFloat(SliderFloatParameters const& parameters, float* value, bool* enabled)
{
    return BasicSlider(parameters, value, enabled);
}

bool AlienImGui::SliderInt(SliderIntParameters const& parameters, int* value, bool* enabled)
{
    return BasicSlider(parameters, value, enabled);
}

bool AlienImGui::SliderFloat2(SliderFloat2Parameters const& parameters, float& valueX, float& valueY)
{
    ImGui::PushID(parameters._name.c_str());

    auto mousePickerButtonSize = parameters._getMousePickerEnabledFunc ? scale(20.0f) + ImGui::GetStyle().FramePadding.x * 2 : 0.0f;
    auto sliderWidth = (ImGui::GetContentRegionAvail().x - scale(parameters._textWidth) - mousePickerButtonSize) / 2 - ImGui::GetStyle().FramePadding.x;
    ImGui::SetNextItemWidth(sliderWidth);
    bool result = ImGui::SliderFloat("##sliderX", &valueX, parameters._min.x, parameters._max.x, parameters._format.c_str(), 0);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(sliderWidth);
    result |= ImGui::SliderFloat("##sliderY", &valueY, parameters._min.y, parameters._max.y, parameters._format.c_str(), 0);

    //mouse picker
    if (parameters._getMousePickerEnabledFunc) {
        ImGui::SameLine();
        auto mousePickerEnabled = parameters._getMousePickerEnabledFunc.value()();
        if (AlienImGui::SelectableButton(AlienImGui::SelectableButtonParameters().name(ICON_FA_CROSSHAIRS), mousePickerEnabled)) {
            parameters._setMousePickerEnabledFunc.value()(mousePickerEnabled);
        }
        AlienImGui::Tooltip("Select a position with the mouse");
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
        if (AlienImGui::revertButton(parameters._name)) {
            valueX = parameters._defaultValue->x;
            valueY = parameters._defaultValue->y;
        }
        ImGui::EndDisabled();
    }

    //text
    if (!parameters._name.empty()) {
        ImGui::SameLine();
        ImGui::TextUnformatted(parameters._name.c_str());
    }

    //tooltip
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    ImGui::PopID();
    return result;
}

void AlienImGui::SliderInputFloat(SliderInputFloatParameters const& parameters, float& value)
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
    ImGui::TextUnformatted(parameters._name.c_str());
}

bool AlienImGui::InputInt(InputIntParameters const& parameters, int& value, bool* enabled)
{
    auto textWidth = scale(parameters._textWidth);
    auto infinityButtonWidth = scale(30);
    auto isInfinity = value == std::numeric_limits<int>::max();
    auto showInfinity = parameters._infinity && (!parameters._readOnly || isInfinity);

    if (enabled) {
        ImGui::Checkbox(("##checkbox" + parameters._name).c_str(), enabled);
        if (!(*enabled) && parameters._disabledValue) {
            value = *parameters._disabledValue;
        }
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    auto inputWidth = ImGui::GetContentRegionAvail().x - textWidth;
    if (showInfinity) {
        inputWidth -= infinityButtonWidth + ImGui::GetStyle().FramePadding.x;
    }

    auto result = false;
    if (!isInfinity) {
        ImGui::SetNextItemWidth(inputWidth);
        ImGuiInputTextFlags flags = parameters._readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
        result = ImGui::InputInt(("##" + parameters._name).c_str(), &value, 1, 100, flags);
    } else {
        std::string text = "infinity";
        result = InputText(InputTextParameters().readOnly(true).width(inputWidth).textWidth(0), text);
    }
    if (parameters._defaultValue) {
        ImGui::SameLine();
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }
    if (showInfinity) {
        ImGui::SameLine();
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
    if (enabled && parameters._keepTextEnabled) {
        ImGui::EndDisabled();
    }
    AlienImGui::Text(parameters._name);
    if (enabled && !parameters._keepTextEnabled) {
        ImGui::EndDisabled();
    }
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }
    return result;
}

namespace
{
    template <typename Parameters, typename T, typename Callable>
    bool optionalWidgetAdaptor(Parameters parameters, std::optional<T>& optionalValue, T const& defaultValue, Callable const& func)
    {
        auto newParameters = parameters;
        newParameters.keepTextEnabled(true);

        auto enabled = optionalValue.has_value();
        auto value = optionalValue.value_or(parameters._defaultValue.value_or(defaultValue));
        auto result = func(newParameters, value, &enabled);
        result |= (optionalValue.has_value() != enabled);
        optionalValue = enabled ? std::make_optional(value) : std::nullopt;
        return result;
    }
}

bool AlienImGui::InputOptionalInt(InputIntParameters const& parameters, std::optional<int>& optValue)
{
    return optionalWidgetAdaptor(parameters, optValue, 0, &AlienImGui::InputInt);
}

bool AlienImGui::InputFloat(InputFloatParameters const& parameters, float& value)
{
    auto textWidth = StyleRepository::get().scale(parameters._textWidth);

    ImGuiInputTextFlags flags = parameters._readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    auto result = ImGui::InputFloat(("##" + parameters._name).c_str(), &value, parameters._step, 0, parameters._format.c_str(), flags);
    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

    if (parameters._tooltip) {
        HelpMarker(*parameters._tooltip);
    }
    return result;
}

void AlienImGui::InputFloat2(InputFloat2Parameters const& parameters, float& value1, float& value2)
{
    auto textWidth = StyleRepository::get().scale(parameters._textWidth);

    ImGuiInputTextFlags flags = parameters._readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    static float value[2];
    value[0] = value1;
    value[1] = value2;
    ImGui::InputFloat2(("##" + parameters._name).c_str(), value, parameters._format.c_str(), flags);
    value1 = value[0];
    value2 = value[1];
    ImGui::SameLine();
    if (parameters._defaultValue1 && parameters._defaultValue2) {
        ImGui::BeginDisabled(value1 == *parameters._defaultValue1 && value2 == *parameters._defaultValue2);
        if (revertButton(parameters._name)) {
            value1 = *parameters._defaultValue1;
            value2 = *parameters._defaultValue2;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

    if (parameters._tooltip) {
        HelpMarker(*parameters._tooltip);
    }
}

bool AlienImGui::ColorField(uint32_t cellColor, float width, float height)
{
    if (width == 0) {
        width = 30.0f;
    }

    if (height == 0) {
        width = ImGui::GetTextLineHeight() + ImGui::GetStyle().FramePadding.y * 2;
    }

    float h, s, v;
    AlienImGui::ConvertRGBtoHSV(cellColor, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s, v));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, s, v));
    auto result = ImGui::Button("##button", ImVec2(scale(width), scale(height)));
    ImGui::PopStyleColor(3);

    return result;
}

void AlienImGui::CheckboxColorMatrix(CheckboxColorMatrixParameters const& parameters, bool (&value)[MAX_COLORS][MAX_COLORS])
{
    BasicInputColorMatrixParameters<bool> basicParameters;
    basicParameters._name = parameters._name;
    basicParameters._textWidth = parameters._textWidth;
    basicParameters._defaultValue = parameters._defaultValue;
    basicParameters._tooltip = parameters._tooltip;
    BasicInputColorMatrix<bool>(basicParameters, value);
}

void AlienImGui::InputIntColorMatrix(InputIntColorMatrixParameters const& parameters, int (&value)[MAX_COLORS][MAX_COLORS])
{
    BasicInputColorMatrixParameters<int> basicParameters;
    basicParameters._name = parameters._name;
    basicParameters._min = parameters._min;
    basicParameters._max = parameters._max;
    basicParameters._logarithmic = parameters._logarithmic;
    basicParameters._textWidth = parameters._textWidth;
    basicParameters._defaultValue = parameters._defaultValue;
    basicParameters._tooltip = parameters._tooltip;
    BasicInputColorMatrix(basicParameters, value);
}

void AlienImGui::InputFloatColorMatrix(InputFloatColorMatrixParameters const& parameters, float (&value)[MAX_COLORS][MAX_COLORS], bool* enabled)
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
    BasicInputColorMatrix(basicParameters, value, enabled);
}

bool AlienImGui::InputText(InputTextParameters const& parameters, char* buffer, int bufferSize)
{
    auto width = parameters._width != 0.0f ? scale(parameters._width) : ImGui::GetContentRegionAvail().x;
    ImGui::SetNextItemWidth(width - scale(parameters._textWidth));
    if (parameters._monospaceFont) {
        ImGui::PushFont(StyleRepository::get().getMonospaceMediumFont());
    }
    ImGuiInputTextFlags flags = 0;
    if (parameters._readOnly) {
        flags |= ImGuiInputTextFlags_ReadOnly;
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
    if (parameters._monospaceFont) {
        ImGui::PopFont();
    }
    if (parameters._defaultValue) {
        ImGui::SameLine();
        ImGui::BeginDisabled(std::string(buffer) == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            StringHelper::copy(buffer, bufferSize, *parameters._defaultValue);
        }
        ImGui::EndDisabled();
    }
    if (!parameters._name.empty()) {
        ImGui::SameLine();
        ImGui::TextUnformatted(parameters._name.c_str());
    }
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    return result;
}

bool AlienImGui::InputText(InputTextParameters const& parameters, std::string& text)
{
    static char buffer[1024];
    StringHelper::copy(buffer, IM_ARRAYSIZE(buffer), text);
    auto result = InputText(parameters, buffer, IM_ARRAYSIZE(buffer));
    text = std::string(buffer);

    return result;
}

void AlienImGui::InputTextMultiline(InputTextMultilineParameters const& parameters, std::string& text)
{
    static char buffer[1024*16];
    StringHelper::copy(buffer, IM_ARRAYSIZE(buffer), text);

    auto textWidth = StyleRepository::get().scale(parameters._textWidth);
    auto height = parameters._height == 0
        ? ImGui::GetContentRegionAvail().y
        : StyleRepository::get().scale(parameters._height);
    auto id = parameters._hint.empty() ? ("##" + parameters._name).c_str() : ("##" + parameters._hint).c_str();
    ImGui::InputTextEx(
        ("##" + parameters._name).c_str(),
        parameters._hint.c_str(),
        buffer,
        IM_ARRAYSIZE(buffer),
        {ImGui::GetContentRegionAvail().x - textWidth, height},
        ImGuiInputTextFlags_Multiline);
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

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

bool AlienImGui::Combo(ComboParameters& parameters, int& value, bool* enabled)
{
    auto textWidth = StyleRepository::get().scale(parameters._textWidth);

    const char** items = new const char*[parameters._values.size()];
    for (int i = 0; i < parameters._values.size(); ++i) {
        items[i] = parameters._values[i].c_str();
    }

    if (enabled) {
        ImGui::Checkbox(("##checkbox" + parameters._name).c_str(), enabled);
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    auto result = ImGui::Combo(
        ("##" + parameters._name).c_str(),
        &value,
        vectorGetter,
        static_cast<void*>(&parameters._values),
        parameters._values.size());
    ImGui::PopItemWidth();

    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

    if (enabled) {
        ImGui::EndDisabled();
    }

    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    delete[] items;
    return result;
}

bool AlienImGui::Switcher(SwitcherParameters& parameters, int& value, bool* enabled /*= nullptr*/)
{
    ImGui::PushID(parameters._name.c_str());

    //enable button
    if (enabled) {
        ImGui::Checkbox("##checkbox", enabled);
        ImGui::BeginDisabled(!(*enabled));
        ImGui::SameLine();
    }

    static auto constexpr buttonWidth = 20;
    auto width = parameters._width != 0.0f ? scale(parameters._width) : ImGui::GetContentRegionAvail().x;
    auto textAndButtonWidth = scale(parameters._textWidth + buttonWidth * 2) + ImGui::GetStyle().FramePadding.x * 4;
    auto switcherWidth = width - textAndButtonWidth;

    auto result = false;
    auto numValues = toInt(parameters._values.size());

    std::string text = parameters._values[value];

    static char buffer[1024];
    StringHelper::copy(buffer, IM_ARRAYSIZE(buffer), text);

    ImGui::SetNextItemWidth(switcherWidth);
    ImGui::InputText(("##" + parameters._name).c_str(), buffer, IM_ARRAYSIZE(buffer), ImGuiInputTextFlags_ReadOnly);

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_CARET_LEFT, ImVec2(scale(buttonWidth), 0))) {
        value = (value + numValues - 1) % numValues;
        result = true;
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_CARET_RIGHT, ImVec2(scale(buttonWidth), 0))) {
        value = (value + 1) % numValues;
        result = true;
    }

    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

    if (enabled) {
        ImGui::EndDisabled();
    }

    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    ImGui::PopID();
    return result;
}

bool AlienImGui::ComboColor(ComboColorParameters const& parameters, int& value, bool* enabled)
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
    AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[value], h, s, v);
    if (enabled && !(*enabled)) {
        s = 0;
        v = 0.2f;
    }
    ImGui::GetWindowDrawList()->AddRectFilled(
        ImVec2(comboPos.x + style.FramePadding.x, comboPos.y + style.FramePadding.y),
        ImVec2(comboPos.x + style.FramePadding.x + colorFieldWidth2, comboPos.y + style.FramePadding.y + ImGui::GetTextLineHeight()),
        ImColor::HSV(h, s, v));


    if (enabled && parameters._keepTextEnabled) {
        ImGui::EndDisabled();
    }
    AlienImGui::Text(parameters._name);
    if (enabled && !parameters._keepTextEnabled) {
        ImGui::EndDisabled();
    }

    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    return true;
}

bool AlienImGui::ComboOptionalColor(ComboColorParameters const& parameters, std::optional<int>& value)
{
    return optionalWidgetAdaptor(parameters, value, 0, &AlienImGui::ComboColor);
}

void AlienImGui::InputColorTransition(InputColorTransitionParameters const& parameters, int sourceColor, int& targetColor, int& transitionAge)
{
    //source color field
    ImGui::PushID(sourceColor);
    AlienImGui::ColorField(Const::IndividualCellColors[sourceColor]);
    ImGui::SameLine();

    //combo for target color
    AlienImGui::Text(ICON_FA_LONG_ARROW_ALT_RIGHT);
    ImGui::SameLine();
    ImGui::PushID("color");
    AlienImGui::ComboColor(AlienImGui::ComboColorParameters().width(70.0f).textWidth(0), targetColor);
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
        if (revertButton(parameters._name)) {
            transitionAge = *parameters._defaultTransitionAge;
            targetColor = *parameters._defaultTargetColor;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }
    ImGui::PopID();
    ImGui::PopID();
}

bool AlienImGui::Checkbox(CheckboxParameters const& parameters, bool& value)
{
    auto result = ImGui::Checkbox(("##" + parameters._name).c_str(), &value);
    if (parameters._textWidth != 0) {
        ImGui::SameLine();
        ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x - scale(parameters._textWidth), 0.0f));
    }

    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            value = *parameters._defaultValue;
            result = true;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    return result;
}

bool AlienImGui::SelectableButton(SelectableButtonParameters const& parameters, bool& value)
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
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(ImVec2(pos.x - ImGui::GetStyle().FramePadding.x, pos.y));
    auto result = ImGui::Button(parameters._name.c_str(), {parameters._width, 0});
    if (result) {
        value = !value;
    }
    ImGui::PopStyleColor(3);

    return result;
}

void AlienImGui::Text(std::string const& text)
{
    ImGui::TextUnformatted(text.c_str());
}

void AlienImGui::BoldText(std::string const& text)
{
    ImGui::PushFont(StyleRepository::get().getSmallBoldFont());
    AlienImGui::Text(text);
    ImGui::PopFont();
}

void AlienImGui::MonospaceText(std::string const& text)
{
    ImGui::PushFont(StyleRepository::get().getMonospaceMediumFont());
    ImGui::PushStyleColor(ImGuiCol_Text, (ImVec4)Const::MonospaceColor);
    Text(text);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}

bool AlienImGui::BeginMenuButton(std::string const& text, bool& toggle, std::string const& popup, float focus)
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2);
    const auto active = toggle;
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
    if (AlienImGui::Button(text.c_str())) {
        toggle = !toggle;
    }
    if (ImGui::IsItemHovered()) {
        toggle = true;
    }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);

    if (!ImGui::GetIO().WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        toggle = false;
    }

    if (toggle) {
        bool open = true;
        ImVec2 buttonPos{pos.x, pos.y};
        ImVec2 buttonSize = ImGui::GetItemRectSize();

        auto height = ImGui::GetWindowHeight();
        ImVec2 windowPos{pos.x, pos.y + height};
        ImGui::SetNextWindowPos(windowPos);
        if (focus) {
            ImGui::SetNextWindowFocus();
        }
        if (ImGui::Begin(
                popup.c_str(),
                &open,
                ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize)) {

            auto mousePos = ImGui::GetMousePos();
            auto windowSize = ImGui::GetWindowSize();
            if (/*ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||*/
                ((mousePos.x < windowPos.x || mousePos.y < windowPos.y || mousePos.x > windowPos.x + windowSize.x
                    || mousePos.y > windowPos.y + windowSize.y)
                && (mousePos.x < buttonPos.x || mousePos.y < buttonPos.y || mousePos.x > buttonPos.x + buttonSize.x
                    || mousePos.y > buttonPos.y + buttonSize.y))) {
                toggle = false;
                EndMenuButton();
            }
        } else {
            toggle = false;
        }
    }
    return toggle;
}

void AlienImGui::EndMenuButton()
{
    ImGui::End();
}

bool AlienImGui::ShutdownButton()
{
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 7);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ImportantButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ImportantButtonHoveredColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ImportantButtonActiveColor);
    auto result = ImGui::Button(ICON_FA_POWER_OFF);
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);

    return result;
}

void AlienImGui::ColorButtonWithPicker(ColorButtonWithPickerParameters const& parameters, uint32_t& color, uint32_t& backupColor, uint32_t(& savedPalette)[32])
{
    ImVec4 imGuiColor = ImColor(color);
    ImVec4 imGuiBackupColor = ImColor(backupColor);
    ImVec4 imGuiSavedPalette[32];
    for (int i = 0; i < IM_ARRAYSIZE(imGuiSavedPalette); ++i) {
        imGuiSavedPalette[i] = ImColor(savedPalette[i]);
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
    color = static_cast<ImU32>(ImColor(imGuiColor));
    backupColor = static_cast<ImU32>(ImColor(imGuiBackupColor));
    for (int i = 0; i < 32; ++i) {
        savedPalette[i] = static_cast<ImU32>(ImColor(imGuiSavedPalette[i]));
    }

    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(color == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            color = *parameters._defaultValue;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }
}

void AlienImGui::NegativeSpacing()
{
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(ImVec2(pos.x - ImGui::GetStyle().FramePadding.x, pos.y));
}

void AlienImGui::Separator()
{
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();
}

void AlienImGui::MovableSeparator(float& height)
{
    ImGui::Button("##MovableSeparator", ImVec2(-1, scale(5.0f)));
    if (ImGui::IsItemActive()) {
        height -= ImGui::GetIO().MouseDelta.y;
    }
}

void AlienImGui::Group(std::string const& text, std::optional<std::string> const& tooltip)
{
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted(text.c_str());
    if (tooltip.has_value()) {
        AlienImGui::HelpMarker(*tooltip);
    }
    ImGui::Separator();
    ImGui::Spacing();
}

bool AlienImGui::ToolbarButton(std::string const& text)
{
    auto id = std::to_string(ImGui::GetID(text.c_str()));

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
    auto result = ImGui::Button(text.c_str(), {buttonSize, buttonSize});

    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar();
    ImGui::PopFont();
    return result;
}

bool AlienImGui::SelectableToolbarButton(std::string const& text, int& value, int selectionValue, int deselectionValue)
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

void AlienImGui::VerticalSeparator(float length)
{
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    auto cursorPos = ImGui::GetCursorScreenPos();
    auto color = ImColor(ImGui::GetStyle().Colors[ImGuiCol_Border]);
    color.Value.w *= ImGui::GetStyle().Alpha;
    drawList->AddLine(ImVec2(cursorPos.x, cursorPos.y - ImGui::GetStyle().FramePadding.y), ImVec2(cursorPos.x, cursorPos.y + scale(length)), color, 2.0f);
    ImGui::Dummy(ImVec2(ImGui::GetStyle().FramePadding.x * 2, 1));
}

void AlienImGui::ToolbarSeparator()
{
    VerticalSeparator(40.0f);
}

bool AlienImGui::Button(std::string const& text, float size)
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

bool AlienImGui::CollapseButton(bool collapsed)
{
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0, 0, 0, 0));
    auto result = ImGui::ArrowButton("##collapseButton", collapsed ? ImGuiDir_Right : ImGuiDir_Down);
    ImGui::PopStyleColor(1);
    return result;
}

bool AlienImGui::BeginTreeNode(TreeNodeParameters const& parameters)
{
    if (parameters._highlighted) {
        ImGui::PushStyleColor(ImGuiCol_Header, Const::TreeNodeHighlightedColor.Value);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, Const::TreeNodeHighlightedHoveredColor.Value);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, Const::TreeNodeHighlightedActiveColor.Value);
    } else {
        ImGui::PushStyleColor(ImGuiCol_Header, Const::TreeNodeColor.Value);
        ImGui::PushStyleColor(ImGuiCol_HeaderHovered, Const::TreeNodeHoveredColor.Value);
        ImGui::PushStyleColor(ImGuiCol_HeaderActive, Const::TreeNodeActiveColor.Value);
    }
    ImGuiTreeNodeFlags treeNodeClosedFlags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_Framed;
    ImGuiTreeNodeFlags treeNodeOpenFlags = treeNodeClosedFlags | ImGuiTreeNodeFlags_DefaultOpen;
    ImGui::PushFont(StyleRepository::get().getSmallBoldFont());
    bool result = ImGui::TreeNodeEx(parameters._text.c_str(), parameters._defaultOpen ? treeNodeOpenFlags : treeNodeClosedFlags);
    ImGui::PopFont();
    ImGui::PopStyleColor(3);
    return result;
}

void AlienImGui::EndTreeNode()
{
    ImGui::TreePop();
}

bool AlienImGui::Button(ButtonParameters const& parameters)
{
    auto width = ImGui::GetContentRegionAvail().x - StyleRepository::get().scale(parameters._textWidth);
    auto result = ImGui::Button(parameters._buttonText.c_str(), {width, 0});
    ImGui::SameLine();

    if (parameters._showDisabledRevertButton) {
        ImGui::BeginDisabled(true);
        revertButton(parameters._name);
        ImGui::EndDisabled();
        ImGui::SameLine();
    }
    ImGui::TextUnformatted(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }
    return result;
}

void AlienImGui::Spinner(SpinnerParameters const& parameters)
{
    static std::optional<std::chrono::steady_clock::time_point> spinnerRefTimepoint;
    static float spinnerAngle = 0;
    if (!spinnerRefTimepoint.has_value()) {
        spinnerRefTimepoint = std::chrono::steady_clock::now();
    }
    auto now = std::chrono::steady_clock::now();
    auto duration = toFloat(std::chrono::duration_cast<std::chrono::milliseconds>(now - *spinnerRefTimepoint).count());

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    AlienImGui::RotateStart(drawList);
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
    AlienImGui::RotateEnd(spinnerAngle, drawList);
}

void AlienImGui::Tooltip(std::string const& text, bool delay)
{
    if (ImGui::IsItemHovered() && (!delay || (delay && GImGui->HoveredIdTimer > HoveredTimer))) {
        ImGui::BeginTooltip();
        ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::TooltipTextColor);
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::PopStyleColor();
        ImGui::EndTooltip();
    }
}

void AlienImGui::Tooltip(std::function<std::string()> const& textFunc, bool delay)
{
    if (ImGui::IsItemHovered() && (!delay || (delay && GImGui->HoveredIdTimer > HoveredTimer))) {
        Tooltip(textFunc(), delay);
    }
}

void AlienImGui::ConvertRGBtoHSV(uint32_t rgb, float& h, float& s, float& v)
{
    return ImGui::ColorConvertRGBtoHSV(
        toFloat((rgb >> 16) & 0xff) / 255, toFloat((rgb >> 8) & 0xff) / 255, toFloat((rgb & 0xff)) / 255, h, s, v);
}

bool AlienImGui::ToggleButton(ToggleButtonParameters const& parameters, bool& value)
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
    ImGui::TextUnformatted(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    return value != origValue;
}

bool AlienImGui::ShowPreviewDescription(PreviewDescription const& desc, float& zoom, std::optional<int>& selectedNode)
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

    ImGui::SetCursorPos({std::max(0.0f, windowSize.x - previewSize.x) / 2, std::max(0.0f, windowSize.y - previewSize.y) / 2});
    if (ImGui::BeginChild("##genome", ImVec2(previewSize.x, previewSize.y), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        auto windowPos = ImGui::GetWindowPos();
        RealVector2D offset{windowPos.x + cellSize, windowPos.y + cellSize};

        ImGui::SetCursorPos({previewSize.x - 1, previewSize.y - 1});

        //draw cells
        for (auto const& cell : desc.cells) {
            auto cellPos = (cell.pos - upperLeft) * cellSize + offset;
            float h, s, v;
            AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell.color], h, s, v);

            auto cellRadiusFactor = zoom > ZoomLevelForConnections ? 0.25f : 0.5f;
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, cellSize * cellRadiusFactor, ImColor::HSV(h, s * 1.2f, v * 1.0f));

            if (zoom > ZoomLevelForLabels) {
                RealVector2D textPos(cellPos.x - cellSize / 8, cellPos.y - cellSize / 4);
                drawTextWithShadow(std::to_string(cell.executionOrderNumber), textPos.x, textPos.y);
            }

            if (selectedNode && cell.nodeIndex == *selectedNode) {
                if (zoom > ZoomLevelForLabels) {
                    drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor(1.0f, 1.0f, 1.0f));
                } else {
                    drawList->AddCircle({cellPos.x, cellPos.y}, cellSize / 2, ImColor::HSV(h, s * 0.8f, v * 1.2f));
                }
            }

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                auto mousePos = ImGui::GetMousePos();
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

bool AlienImGui::CellFunctionCombo(CellFunctionComboParameters& parameters, int& value)
{
    auto modCellFunctionStrings = Const::CellFunctionStrings;
    auto noneString = modCellFunctionStrings.back();
    modCellFunctionStrings.pop_back();
    modCellFunctionStrings.insert(modCellFunctionStrings.begin(), noneString);

    value = (value + 1) % CellFunction_Count;
    auto result = AlienImGui::Combo(
        AlienImGui::ComboParameters().name(parameters._name).values(modCellFunctionStrings).textWidth(parameters._textWidth).tooltip(parameters._tooltip),
        value);
    value = (value + CellFunction_Count - 1) % CellFunction_Count;
    return result;
}

bool AlienImGui::AngleAlignmentCombo(AngleAlignmentComboParameters& parameters, int& value)
{
    std::vector const AngleAlignmentStrings = {"None"s, "180 deg"s, "120 deg"s, "90 deg"s, "72 deg"s, "60 deg"s};
    return AlienImGui::Combo(
        AlienImGui::ComboParameters().name(parameters._name).values(AngleAlignmentStrings).textWidth(parameters._textWidth).tooltip(parameters._tooltip), value);
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

void AlienImGui::NeuronSelection(
    NeuronSelectionParameters const& parameters,
    std::vector<std::vector<float>>& weights,
    std::vector<float>& biases,
    std::vector<NeuronActivationFunction>& activationFunctions)
{
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
            if (std::abs(weights[j][i]) > NEAR_ZERO) {
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
            if (std::abs(weights[j][i]) <= NEAR_ZERO) {
                continue;
            }
            auto thickness = std::min(4.0f, std::abs(weights[j][i]));
            drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, calcColor(weights[j][i]), thickness);
        }
    }

    //visualize activation functions
    auto calcPlotPosition = [&](RealVector2D const& refPos, float x, NeuronActivationFunction activationFunction) {
        float value = 0;
        switch (activationFunction) {
        case NeuronActivationFunction_Sigmoid:
            value = Math::sigmoid(x);
            break;
        case NeuronActivationFunction_BinaryStep:
            value = Math::binaryStep(x);
            break;
        case NeuronActivationFunction_Identity:
            value = x / 4;
            break;
        case NeuronActivationFunction_Abs:
            value = std::abs(x) / 4;
            break;
        case NeuronActivationFunction_Gaussian:
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
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Activation function")
                    .textWidth(editorColumnTextWidth)
                    .values(Const::ActivationFunctions)
                    .tooltip(Const::GenomeNeuronActivationFunctionTooltip),
                activationFunctions.at(selectedOutput));
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(editorColumnTextWidth).tooltip(Const::GenomeNeuronWeightAndBiasTooltip),
                weights.at(selectedOutput).at(selectedInput));
            table.next();
            AlienImGui::InputFloat(
                AlienImGui::InputFloatParameters().name("Bias").step(0.05f).textWidth(editorColumnTextWidth).tooltip(Const::GenomeNeuronWeightAndBiasTooltip),
                biases.at(selectedOutput));
            table.end();
        }
        if (AlienImGui::Button("Clear")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i][j] = 0;
                }
                biases[i] = 0;
                activationFunctions[i] = NeuronActivationFunction_Sigmoid;
            }
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Identity")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i][j] = i == j ? 1.0f : 0.0f;
                }
                biases[i] = 0.0f;
                activationFunctions[i] = NeuronActivationFunction_Identity;
            }
        }
        ImGui::SameLine();
        if (AlienImGui::Button("Randomize")) {
            for (int i = 0; i < MAX_CHANNELS; ++i) {
                for (int j = 0; j < MAX_CHANNELS; ++j) {
                    weights[i][j] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                }
                biases[i] = NumberGenerator::get().getRandomFloat(-4.0f, 4.0f);
                activationFunctions[i] = NumberGenerator::get().getRandomInt(NeuronActivationFunction_Count);
            }
        }
    }
    ImGui::EndChild();
}

void AlienImGui::OnlineSymbol()
{
    auto counter = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    counter = (((counter % 2000) + 2000) % 2000);
    auto color = ImColor::HSV(0.0f, counter < 1000 ? toFloat(counter) / 1000.0f : 2.0f - toFloat(counter) / 1000.0f, 1.0f);
    ImGui::PushStyleColor(ImGuiCol_Text, color.Value);
    ImGui::Text(ICON_FA_GENDERLESS);
    ImGui::PopStyleColor();
}

void AlienImGui::LastDayOnlineSymbol()
{
    auto color = ImColor::HSV(0.16f, 0.5f, 0.66f);
    ImGui::PushStyleColor(ImGuiCol_Text, color.Value);
    ImGui::Text(ICON_FA_GENDERLESS);
    ImGui::PopStyleColor();
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
bool AlienImGui::BasicSlider(Parameter const& parameters, T* value, bool* enabled)
{
    ImGui::PushID(parameters._name.c_str());

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
        ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - scale(parameters._textWidth));
        if (parameters._colorDependence && isExpanded) {
            AlienImGui::ColorField(Const::IndividualCellColors[color], 0);
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
                if (revertButton(parameters._name)) {
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
                ImGui::TextUnformatted(parameters._name.c_str());
            }

            //tooltip
            if (parameters._tooltip) {
                if (enabled) {
                    ImGui::EndDisabled();
                }
                AlienImGui::HelpMarker(*parameters._tooltip);
                if (enabled) {
                    ImGui::BeginDisabled(!(*enabled));
                }
            }
        }
    }
    if (enabled) {
        ImGui::EndDisabled();
    }
    ImGui::PopID();
    return result;
}

template <typename T>
void AlienImGui::BasicInputColorMatrix(BasicInputColorMatrixParameters<T> const& parameters, T (&value)[MAX_COLORS][MAX_COLORS], bool* enabled)
{
    ImGui::PushID(parameters._name.c_str());

    if (enabled) {
        ImGui::Checkbox("##cellFunctionAttackerGenomeComplexityBonus", enabled);
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
        if (revertButton(parameters._name)) {
            for (int row = 0; row < MAX_COLORS; ++row) {
                for (int col = 0; col < MAX_COLORS; ++col) {
                    value[row][col] = (*parameters._defaultValue)[row][col];
                }
            }
        }
        ImGui::EndDisabled();
    }

    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

    if (enabled) {
        ImGui::EndDisabled();
    }

    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    ImGui::PopID();
}

//RotateStart, RotationCenter, etc. are taken from https://gist.github.com/carasuca/e72aacadcf6cf8139de46f97158f790f
//>>>>>>>>>>
void AlienImGui::RotateStart(ImDrawList* drawList)
{
    _rotationStartIndex = drawList->VtxBuffer.Size;
}

ImVec2 AlienImGui::RotationCenter(ImDrawList* drawList)
{
    ImVec2 l(FLT_MAX, FLT_MAX), u(-FLT_MAX, -FLT_MAX);  // bounds

    const auto& buf = drawList->VtxBuffer;
    for (int i = _rotationStartIndex; i < buf.Size; i++)
        l = ImMin(l, buf[i].pos), u = ImMax(u, buf[i].pos);

    return ImVec2((l.x + u.x) / 2, (l.y + u.y) / 2);  // or use _ClipRectStack?
}

bool AlienImGui::revertButton(std::string const& id)
{
    auto result = ImGui::Button((ICON_FA_UNDO "##" + id).c_str());
    AlienImGui::Tooltip("Revert changes");
    return result;
}

namespace
{
    ImVec2 operator-(const ImVec2& l, const ImVec2& r)
    {
        return {l.x - r.x, l.y - r.y};
    }
}

void AlienImGui::RotateEnd(float angle, ImDrawList* drawList)
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

int AlienImGui::DynamicTableLayout::calcNumColumns(float tableWidth, float columnWidth)
{
    return std::max(toInt(tableWidth / scale(columnWidth)), 1);
}

AlienImGui::DynamicTableLayout::DynamicTableLayout(float columnWidth)
    : _columnWidth(columnWidth)
{
    _numColumns = calcNumColumns(ImGui::GetContentRegionAvail().x, columnWidth);
}

bool AlienImGui::DynamicTableLayout::begin()
{
    auto result = ImGui::BeginTable("##", _numColumns, ImGuiTableFlags_None);
    if (result) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
    }
    return result;
}
void AlienImGui::DynamicTableLayout::end()
{
    ImGui::EndTable();
}

void AlienImGui::DynamicTableLayout::next()
{
    auto currentCol = (++_elementNumber) % _numColumns;
    if (currentCol > 0) {
        ImGui::TableSetColumnIndex(currentCol);
        AlienImGui::VerticalSeparator();
        ImGui::SameLine();
    } else {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
    }
}
