#include "AlienImGui.h"

#include <imgui.h>
#include <imgui_internal.h>

#include "Fonts/IconsFontAwesome5.h"

#include "Base/StringHelper.h"
#include "EngineInterface/Colors.h"
#include "EngineInterface/Constants.h"

#include "CellFunctionStrings.h"
#include "StyleRepository.h"

namespace
{
    bool revertButton(std::string const& id)
    {
        return ImGui::Button((ICON_FA_UNDO "##" + id).c_str());
    }
}

void AlienImGui::HelpMarker(std::string const& text)
{
    ImGui::SameLine();
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextInfoColor);
    ImGui::Text("(?)");
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

bool AlienImGui::SliderFloat(SliderFloatParameters const& parameters, float& value)
{
    auto width = StyleRepository::getInstance().scaleContent(parameters._textWidth);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - width);
    auto result = ImGui::SliderFloat(
        ("##" + parameters._name).c_str(),
        &value,
        parameters._min,
        parameters._max,
        parameters._format.c_str(),
        parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
    if (parameters._defaultValue) {
        ImGui::SameLine();
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

bool AlienImGui::SliderInt(SliderIntParameters const& parameters, int& value)
{
    auto width = StyleRepository::getInstance().scaleContent(parameters._textWidth);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - width);
    auto result = ImGui::SliderInt(
        ("##" + parameters._name).c_str(), &value, parameters._min, parameters._max, "%d", parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
    if (parameters._defaultValue) {
        ImGui::SameLine();
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

void AlienImGui::SliderInputFloat(SliderInputFloatParameters const& parameters, float& value)
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);
    auto inputWidth = StyleRepository::getInstance().scaleContent(parameters._inputWidth);

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

bool AlienImGui::InputInt(InputIntParameters const& parameters, int& value)
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    auto result = ImGui::InputInt(("##" + parameters._name).c_str(), &value);
    if (parameters._defaultValue) {
        ImGui::SameLine();
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

void AlienImGui::InputFloat(InputFloatParameters const& parameters, float& value)
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);

    ImGuiInputTextFlags flags = parameters._readOnly ? ImGuiInputTextFlags_ReadOnly : ImGuiInputTextFlags_None;
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    ImGui::InputFloat(("##" + parameters._name).c_str(), &value, parameters._step, 0, parameters._format.c_str(), flags);
    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(value == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            value = *parameters._defaultValue;
        }
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());

    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }
}

void AlienImGui::InputFloatVector(InputFloatVectorParameters const& parameters, std::vector<float>& value)
{
    std::vector<std::vector<float>> wrappedValue{value};
    InputFloatMatrix(
        InputFloatMatrixParameters().name(parameters._name).textWidth(parameters._textWidth).step(parameters._step).format(parameters._format), wrappedValue);
    value = wrappedValue.front();
}

void AlienImGui::InputFloatMatrix(InputFloatMatrixParameters const& parameters, std::vector<std::vector<float>>& value)
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);
    auto rows = value.size();
    auto cols = value.front().size();
    if (ImGui::BeginTable(("##" + parameters._name).c_str(), cols, 0, ImVec2(ImGui::GetContentRegionAvail().x - textWidth, 0))) {
        for (int row = 0; row < rows; ++row) {
            ImGui::PushID(row);
            for (int col = 0; col < cols; ++col) {
                ImGui::PushID(col);
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                ImGui::InputFloat(("##" + parameters._name).c_str(), &value.at(row).at(col), parameters._step, 0, parameters._format.c_str());
                ImGui::PopID();
            }
            ImGui::TableNextRow();
            ImGui::PopID();
        }
        ImGui::EndTable();

        ImGui::SameLine();
        ImGui::TextUnformatted(parameters._name.c_str());
    }
}

bool AlienImGui::ColorField(uint32_t cellColor, int width/* = -1*/)
{
    if (width == 0) {
        width = StyleRepository::getInstance().scaleContent(30);
    }
    float h, s, v;
    AlienImGui::ConvertRGBtoHSV(cellColor, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.7f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.7f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.7f));
/*
    ImVec2 pos = ImGui::GetCursorScreenPos();
    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + ImGui::GetStyle().FramePadding.y));
*/
    auto result = ImGui::Button("##", ImVec2(width, ImGui::GetTextLineHeight()));
    ImGui::PopStyleColor(3);

    return result;
}

void AlienImGui::InputColorMatrix(InputColorMatrixParameters const& parameters, float (&value)[MAX_COLORS][MAX_COLORS])
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);

    if (ImGui::BeginTable(("##" + parameters._name).c_str(), MAX_COLORS + 1, 0, ImVec2(ImGui::GetContentRegionAvail().x - textWidth, 0))) {
        for (int row = 0; row < MAX_COLORS + 1; ++row) {
            ImGui::PushID(row);
            for (int col = 0; col < MAX_COLORS + 1; ++col) {
                ImGui::PushID(col);
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                if (row == 0 && col > 0) {
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + ImGui::GetStyle().FramePadding.y));
                    ColorField(Const::IndividualCellColors[col - 1], -1);
                } else if (row > 0 && col == 0) {
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + ImGui::GetStyle().FramePadding.y));
                    ColorField(Const::IndividualCellColors[row - 1], -1);
                } else if (row > 0 && col > 0) {
                    ImGui::InputFloat(("##" + parameters._name).c_str(), &value[row - 1][col - 1], 0, 0, parameters._format.c_str());
                }
                ImGui::PopID();
            }
            ImGui::TableNextRow();
            ImGui::PopID();
        }
        ImGui::EndTable();
        ImGui::SameLine();
        if (parameters._defaultValue) {
            bool changed = false;
            for (int row = 0; row < MAX_COLORS; ++row) {
                for (int col = 0; col < MAX_COLORS; ++col) {
                    if(value[row][col] != (*parameters._defaultValue)[row][col]) {
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

        if (parameters._tooltip) {
            AlienImGui::HelpMarker(*parameters._tooltip);
        }
    }
}

void AlienImGui::InputColorVector(InputColorVectorParameters const& parameters, float (&value)[MAX_COLORS])
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);

    if (ImGui::BeginTable(("##" + parameters._name).c_str(), MAX_COLORS, 0, ImVec2(ImGui::GetContentRegionAvail().x - textWidth, 0))) {
        for (int row = 0; row < 2; ++row) {
            ImGui::PushID(row);
            for (int col = 0; col < MAX_COLORS; ++col) {
                ImGui::PushID(col);
                ImGui::TableNextColumn();
                ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
                if (row == 0) {
                    ImVec2 pos = ImGui::GetCursorScreenPos();
                    ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + ImGui::GetStyle().FramePadding.y));
                    ColorField(Const::IndividualCellColors[col], -1);
                } else {
                    ImGui::InputFloat(("##" + parameters._name).c_str(), &value[col], 0, 0, parameters._format.c_str());
                }
                ImGui::PopID();
            }
            ImGui::TableNextRow();
            ImGui::PopID();
        }
        ImGui::EndTable();
        ImGui::SameLine();
        if (parameters._defaultValue) {
            bool changed = false;
            for (int col = 0; col < MAX_COLORS; ++col) {
                if (value[col] != (*parameters._defaultValue)[col]) {
                    changed = true;
                }
            }
            ImGui::BeginDisabled(!changed);
            if (revertButton(parameters._name)) {
                for (int col = 0; col < MAX_COLORS; ++col) {
                    value[col] = (*parameters._defaultValue)[col];
                }
            }
            ImGui::EndDisabled();
        }

        ImGui::SameLine();
        ImGui::TextUnformatted(parameters._name.c_str());

        if (parameters._tooltip) {
            AlienImGui::HelpMarker(*parameters._tooltip);
        }
    }
}

bool AlienImGui::InputText(InputTextParameters const& parameters, char* buffer, int bufferSize)
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - textWidth);
    if (parameters._monospaceFont) {
        ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
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
    ImGui::SameLine();
    if (parameters._defaultValue) {
        ImGui::BeginDisabled(std::string(buffer) == *parameters._defaultValue);
        if (revertButton(parameters._name)) {
            StringHelper::copy(buffer, bufferSize, *parameters._defaultValue);
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

    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);
    auto height = parameters._height == 0
        ? ImGui::GetContentRegionAvail().y
        : StyleRepository::getInstance().scaleContent(parameters._height);
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

bool AlienImGui::Combo(ComboParameters& parameters, int& value)
{
    auto textWidth = StyleRepository::getInstance().scaleContent(parameters._textWidth);

    const char** items = new const char*[parameters._values.size()];
    for (int i = 0; i < parameters._values.size(); ++i) {
        items[i] = parameters._values[i].c_str();
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
    delete[] items;

    return result;
}

bool AlienImGui::ComboColor(ComboColorParameters const& parameters, int& value)
{
    auto& styleRep = StyleRepository::getInstance();
    auto textWidth = styleRep.scaleContent(parameters._textWidth);
    auto comboWidth = !parameters._name.empty() ? ImGui::GetContentRegionAvail().x - textWidth : styleRep.scaleContent(70);
    auto colorFieldWidth1 = comboWidth - styleRep.scaleContent(40);
    auto colorFieldWidth2 = comboWidth - styleRep.scaleContent(30);

    const char* items[] = { "##1", "##2", "##3", "##4", "##5", "##6", "##7" };

    ImVec2 comboPos = ImGui::GetCursorPos();

    ImGui::SetNextItemWidth(comboWidth);
    if (ImGui::BeginCombo(("##" + parameters._name).c_str(), "")) {
        for (int n = 0; n < MAX_COLORS; ++n) {
            bool isSelected = (value == n);

            if (ImGui::Selectable(items[n], isSelected)) {
                value = n;
            }
            ImGui::SameLine();
            ColorField(Const::IndividualCellColors[n], colorFieldWidth1);
            ImGui::SameLine();
            ImGui::TextUnformatted(" ");
            if (isSelected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImVec2 backupPos = ImGui::GetCursorPos();

    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::SetCursorPos(ImVec2(comboPos.x + style.FramePadding.x, comboPos.y + style.FramePadding.y));
    ColorField(Const::IndividualCellColors[value], colorFieldWidth2);

    ImGui::SetCursorPos({backupPos.x, backupPos.y + style.FramePadding.y});

    AlienImGui::Text(parameters._name);
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(0, ImGui::GetTextLineHeight() + style.FramePadding.y));

    return true;
}

void AlienImGui::InputColorTransition(InputColorTransitionParameters const& parameters, int sourceColor, int& targetColor, int& transitionAge)
{
    //source color field
    ImGui::PushID(sourceColor);
    {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y + ImGui::GetStyle().FramePadding.y));
    }
    AlienImGui::ColorField(Const::IndividualCellColors[sourceColor], 0);
    ImGui::SameLine();

    //combo for target color
    {
        ImVec2 pos = ImGui::GetCursorScreenPos();
        ImGui::SetCursorScreenPos(ImVec2(pos.x, pos.y - ImGui::GetStyle().FramePadding.y));
    }
    AlienImGui::Text(ICON_FA_LONG_ARROW_ALT_RIGHT);
    ImGui::SameLine();
    ImGui::PushID(1);
    AlienImGui::ComboColor(AlienImGui::ComboColorParameters(), targetColor);
    ImGui::PopID();

    ImGui::SameLine();
    ImVec2 pos = ImGui::GetCursorPos();
    ImGui::SetCursorPos({pos.x, pos.y - ImGui::GetStyle().FramePadding.y});

    //slider for transition age
    ImGui::PushID(2);
    auto width = StyleRepository::getInstance().scaleContent(parameters._textWidth);

    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x - width);
    ImGui::SliderInt(
        ("##" + parameters._name).c_str(), &transitionAge, parameters._min, parameters._max, "%d", parameters._logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
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
    ImGui::SameLine();
    if (parameters._textWidth != 0) {
        ImGui::Dummy(ImVec2(ImGui::GetContentRegionAvail().x - parameters._textWidth, 0.0f));
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

void AlienImGui::Text(std::string const& text)
{
    ImGui::TextUnformatted(text.c_str());
}

void AlienImGui::MonospaceText(std::string const& text)
{
    ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
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
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::ShutdownButtonColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::ShutdownButtonHoveredColor);
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::ShutdownButtonActiveColor);
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
        {ImGui::GetContentRegionAvail().x - StyleRepository::getInstance().scaleContent(parameters._textWidth), 0});
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

void AlienImGui::Separator()
{
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();
}

void AlienImGui::Group(std::string const& text)
{
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextUnformatted(text.c_str());
    ImGui::Separator();
    ImGui::Spacing();
}

bool AlienImGui::ToolbarButton(std::string const& text)
{
    ImGui::PushFont(StyleRepository::getInstance().getIconFont());
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, {0.5f, 0.75f});
    auto color = Const::ToolbarButtonColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);
    ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(h, s * 0.6f, v * 0.3f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(h, s * 0.7f, v * 0.4f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(h, s * 0.8f, v * 0.6f));
    ImGui::PushStyleColor(ImGuiCol_Text, (ImU32)Const::ToolbarButtonColor);
    auto buttonSize = StyleRepository::getInstance().scaleContent(40.0f);
    auto result = ImGui::Button(text.c_str(), {buttonSize, buttonSize});
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar();
    ImGui::PopFont();
    return result;
}

bool AlienImGui::Button(std::string const& text)
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
    auto result = ImGui::Button(text.c_str());
/*
    ImGui::PopStyleColor(4);
*/
    return result;
}

bool AlienImGui::Button(ButtonParameters const& parameters)
{
    auto width = ImGui::GetContentRegionAvail().x - StyleRepository::getInstance().scaleContent(parameters._textWidth);
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

void AlienImGui::Tooltip(std::string const& text)
{
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(text.c_str());
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void AlienImGui::Tooltip(std::function<std::string()> const& textFunc)
{
    if (ImGui::IsItemHovered()) {
        Tooltip(textFunc());
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
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    float height = ImGui::GetFrameHeight();
    float width = height * 1.55f;
    float radius = height * 0.50f*0.8f;
    height = height * 0.8f;

    ImGui::InvisibleButton(parameters._name.c_str(), ImVec2(width, height));
    if (ImGui::IsItemClicked()) {
        value = !value;
    }

    auto color = Const::ToggleButtonColor;
    float h, s, v;
    ImGui::ColorConvertRGBtoHSV(color.Value.x, color.Value.y, color.Value.z, h, s, v);

    if (ImGui::IsItemHovered()) {
        draw_list->AddRectFilled(
            p,
            ImVec2(p.x + width, p.y + height),
            ImGui::GetColorU32(value ? (ImU32)ImColor::HSV(h, s * 0.9f, v * 0.8f) : (ImU32)ImColor::HSV(h, s * 0.9f, v * 0.4f)),
            height * 0.5f);
    } else {
        draw_list->AddRectFilled(
            p,
            ImVec2(p.x + width, p.y + height),
            ImGui::GetColorU32(value ? (ImU32)ImColor::HSV(h, s * 0.6f, v * 0.7f) : (ImU32)ImColor::HSV(h, s * 0.6f, v * 0.3f)),
            height * 0.50f);
    }
    draw_list->AddCircleFilled(ImVec2(p.x + radius + (value ? 1 : 0) * (width - radius * 2.0f), p.y + radius), radius - 1.5f, IM_COL32(20, 20, 20, 255));
    draw_list->AddCircleFilled(ImVec2(p.x + radius + (value ? 1 : 0) * (width - radius * 2.0f), p.y + radius), radius - 2.5f, IM_COL32(255, 255, 255, 255));

    ImGui::SameLine();
    ImGui::TextUnformatted(parameters._name.c_str());
    if (parameters._tooltip) {
        AlienImGui::HelpMarker(*parameters._tooltip);
    }

    return value != origValue;
}

void AlienImGui::ShowPreviewDescription(PreviewDescription const& desc)
{
    auto const CellSize = StyleRepository::getInstance().scaleContent(20.0f);

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
    RealVector2D previewSize = (lowerRight - upperLeft) * CellSize + RealVector2D(CellSize, CellSize) * 2;

    auto windowSize = ImGui::GetWindowSize();
    ImGui::SetCursorPos({std::max(0.0f, windowSize.x - previewSize.x) / 2, std::max(0.0f, windowSize.y - previewSize.y) / 2});

    if (ImGui::BeginChild("##", ImVec2(previewSize.x, previewSize.y), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        auto windowPos = ImGui::GetWindowPos();
        RealVector2D offset{windowPos.x + CellSize, windowPos.y + CellSize};

        ImGui::SetCursorPos({previewSize.x, previewSize.y});

        for (auto const& connection : desc.connections) {
            auto startPos = (connection.cell1 - upperLeft) * CellSize + offset;
            auto endPos = (connection.cell2 - upperLeft) * CellSize + offset;
            drawList->AddLine({startPos.x, startPos.y}, {endPos.x, endPos.y}, ImColor(1.0f, 1.0f, 1.0f), 2.0f);
        }
        for (auto const& cell : desc.cells) {
            auto cellPos = (cell.pos - upperLeft) * CellSize + offset;
            float h, s, v;
            AlienImGui::ConvertRGBtoHSV(Const::IndividualCellColors[cell.color], h, s, v);
            drawList->AddCircleFilled({cellPos.x, cellPos.y}, CellSize / 4, ImColor::HSV(h, s * 0.7f, v * 0.7f));
            if (cell.selected) {
                drawList->AddCircle({cellPos.x, cellPos.y}, CellSize / 2, ImColor(1.0f, 1.0f, 1.0f));
            }
        }
    }
    ImGui::EndChild();
}

bool AlienImGui::CellFunctionCombo(CellFunctionComboParameters& parameters, int& value)
{
    auto modCellFunctionStrings = Const::CellFunctionStrings;
    auto noneString = modCellFunctionStrings.back();
    modCellFunctionStrings.pop_back();
    modCellFunctionStrings.insert(modCellFunctionStrings.begin(), noneString);

    value = (value + 1) % CellFunction_Count;
    auto result =
        AlienImGui::Combo(AlienImGui::ComboParameters().name(parameters._name).values(modCellFunctionStrings).textWidth(parameters._textWidth), value);
    value = (value + CellFunction_Count - 1) % CellFunction_Count;
    return result;
}

bool AlienImGui::AngleAlignmentCombo(AngleAlignmentComboParameters& parameters, int& value)
{
    std::vector const AngleAlignmentStrings = {"None"s, "Align to 180 deg"s, "Align to 120 deg"s, "Align to 90 deg"s, "Align to 72 deg"s, "Align to 60 deg"s};
    return AlienImGui::Combo(AlienImGui::ComboParameters().name(parameters._name).values(AngleAlignmentStrings).textWidth(parameters._textWidth), value);
}

void AlienImGui::NeuronSelection(
    NeuronSelectionParameters const& parameters,
    std::vector<std::vector<float>> const& weights,
    std::vector<float> const& bias,
    int& selectedInput,
    int& selectedOutput)
{
    auto setDefaultColors = [] {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::NeuronChannelButtonColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::NeuronChannelHoveredColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::NeuronChannelHoveredColor);
    };
    auto setHightlightingColors = [] {
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)Const::NeuronChannelActiveColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)Const::NeuronChannelActiveColor);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)Const::NeuronChannelActiveColor);
    };
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    auto windowPos = ImGui::GetWindowPos();
    auto outputButtonPositionFromRight = StyleRepository::getInstance().scaleContent(parameters._outputButtonPositionFromRight);
    RealVector2D inputPos[MAX_CHANNELS];
    RealVector2D outputPos[MAX_CHANNELS];
    auto biasFieldWidth = ImGui::GetStyle().FramePadding.x * 2;

    //draw buttons and save positions to visualize weights
    for (int i = 0; i < MAX_CHANNELS; ++i) {

        auto startButtonPos = ImGui::GetCursorPos();

        i == selectedInput ? setHightlightingColors() : setDefaultColors();
        if (ImGui::Button(("Input #" + std::to_string(i)).c_str())) {
            selectedInput = i;
        }
        ImGui::PopStyleColor(3);
        if (i == 0) {
            Tooltip("Used by\n" ICON_FA_CARET_RIGHT " Neuron: output of other neuron cell\n" ICON_FA_CARET_RIGHT
                    " Constructor: 0 = construction failed, 1 = construction successful\n" ICON_FA_CARET_RIGHT " Sensor: 0 = nothing found, 1 = region found");
        }
        if (i == 1) {
            Tooltip("Used by\n" ICON_FA_CARET_RIGHT " Neuron: output of other neuron cell\n" ICON_FA_CARET_RIGHT " Sensor: cell density of found region");
        }
        if (i == 2) {
            Tooltip("Used by\n" ICON_FA_CARET_RIGHT " Neuron: output of other neuron cell\n" ICON_FA_CARET_RIGHT " Sensor: distance to found region");
        }
        if (i == 3) {
            Tooltip("Used by\n" ICON_FA_CARET_RIGHT " Neuron: output of other neuron cell\n" ICON_FA_CARET_RIGHT
                    " Sensor: relative angle of found region (when 'scan specific direction' is activated)");
        }
        if (i > 3) {
            Tooltip("Used by\n" ICON_FA_CARET_RIGHT " Neuron: output of other neuron cell");
        }

        auto buttonSize = ImGui::GetItemRectSize();
        inputPos[i] = RealVector2D(
            windowPos.x - ImGui::GetScrollX() + startButtonPos.x + buttonSize.x, windowPos.y - ImGui::GetScrollY() + startButtonPos.y + buttonSize.y / 2);

        ImGui::SameLine(0, ImGui::GetContentRegionAvail().x - buttonSize.x - outputButtonPositionFromRight + ImGui::GetStyle().FramePadding.x);
        startButtonPos = ImGui::GetCursorPos();
        outputPos[i] = RealVector2D(
            windowPos.x - ImGui::GetScrollX() + startButtonPos.x - biasFieldWidth, windowPos.y - ImGui::GetScrollY() + startButtonPos.y + buttonSize.y / 2);

        i == selectedOutput ? setHightlightingColors() : setDefaultColors();
        if (ImGui::Button(("Output #" + std::to_string(i)).c_str())) {
            selectedOutput = i;
        }
        ImGui::PopStyleColor(3);
        if (i == 0) {
            Tooltip("Used by\n" ICON_FA_CARET_RIGHT " Neuron: input to other neuron cell\n" ICON_FA_CARET_RIGHT
                    " Constructor: = 0 means do nothing, abs(*) > threshold means try construct cell\n" ICON_FA_CARET_RIGHT
                    " Attacker: = 0 means do nothing, abs(*) > threshold means try attack nearby cells\n" ICON_FA_CARET_RIGHT
                    " Sensor: = 0 means do nothing, abs(*) > threshold means scan vicinity for cells\n" ICON_FA_CARET_RIGHT
                    " Injector: = 0 means do nothing, abs(*) > threshold means try inject genome to other constructors\n" ICON_FA_CARET_RIGHT
                    " Muscle: abs(*) intensity of muscle process and sign(*) direction of muscle process");
        }
        if (i > 0) {
            Tooltip("Used by\n" ICON_FA_CARET_RIGHT " Neuron: input to other neuron cell");
        }
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        for (int j = 0; j < MAX_CHANNELS; ++j) {
            if (std::abs(weights[j][i]) > NEAR_ZERO) {
                continue;
            }
            drawList->AddLine({inputPos[i].x, inputPos[i].y}, {outputPos[j].x, outputPos[j].y}, ImColor::HSV(0.0f, 0.0f, 0.1f), 2.0f);
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

    //visualize bias
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        drawList->AddRectFilled(
            {outputPos[i].x, outputPos[i].y - biasFieldWidth}, {outputPos[i].x + biasFieldWidth, outputPos[i].y + biasFieldWidth}, calcColor(bias[i]));
    }

    //draw selection
    drawList->AddRectFilled(
        {outputPos[selectedOutput].x, outputPos[selectedOutput].y - biasFieldWidth},
        {outputPos[selectedOutput].x + biasFieldWidth, outputPos[selectedOutput].y + biasFieldWidth},
        ImColor::HSV(0.0f, 0.0f, 1.0f, 0.35f));
    drawList->AddLine(
        {inputPos[selectedInput].x, inputPos[selectedInput].y}, {outputPos[selectedOutput].x, outputPos[selectedOutput].y}, ImColor::HSV(0.0f, 0.0f, 1.0f, 0.35f), 8.0f);
}
