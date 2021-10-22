#include "AlienImGui.h"

#include "imgui.h"

#include "IconFontCppHeaders/IconsFontAwesome5.h"

#include "StyleRepository.h"

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

void AlienImGui::SliderFloat(
    std::string const& name,
    float& value,
    float defaultValue,
    float min,
    float max,
    bool logarithmic,
    std::string const& format,
    boost::optional<std::string> tooltip)
{
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x / 2);
    ImGui::SliderFloat(
        ("##" + name).c_str(), &value, min, max, format.c_str(), logarithmic ? ImGuiSliderFlags_Logarithmic : 0);
    ImGui::SameLine();
    ImGui::BeginDisabled(value == defaultValue);
    if (ImGui::Button((ICON_FA_UNDO "##" + name).c_str())) {
        value = defaultValue;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Text(name.c_str());
    if (tooltip) {
        AlienImGui::HelpMarker(tooltip->c_str());
    }
}

void AlienImGui::SliderInt(
    std::string const& name,
    int& value,
    int defaultValue,
    int min,
    int max,
    boost::optional<std::string> tooltip)
{
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x / 2);
    ImGui::SliderInt(("##" + name).c_str(), &value, min, max);
    ImGui::SameLine();
    ImGui::BeginDisabled(value == defaultValue);
    if (ImGui::Button((ICON_FA_UNDO "##" + name).c_str())) {
        value = defaultValue;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Text(name.c_str());

    if (tooltip) {
        AlienImGui::HelpMarker(tooltip->c_str());
    }
}

void AlienImGui::InputInt(
    std::string const& name,
    int& value,
    int defaultValue,
    boost::optional<std::string> tooltip)
{
    ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x / 2);
    ImGui::InputInt(("##" + name).c_str(), &value);
    ImGui::SameLine();
    ImGui::BeginDisabled(value == defaultValue);
    if (ImGui::Button((ICON_FA_UNDO "##" + name).c_str())) {
        value = defaultValue;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Text(name.c_str());

    if (tooltip) {
        AlienImGui::HelpMarker(tooltip->c_str());
    }
}

void AlienImGui::Combo(std::string const& name, int& value, int defaultValue, std::vector<std::string> const& values)
{
    const char* items[10] = {};
    for(int i = 0; i < values.size(); ++i) {
        items[i] = values[i].c_str();
    }

    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x / 2);
    ImGui::Combo("##", &value, items, toInt(values.size()));
    ImGui::PopItemWidth();

    ImGui::SameLine();
    ImGui::BeginDisabled(value == defaultValue);
    if (ImGui::Button((ICON_FA_UNDO "##" + name).c_str())) {
        value = defaultValue;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::Text(name.c_str());
}

bool AlienImGui::BeginMenuButton(std::string const& text, bool& toggle, std::string const& popup)
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
    if (ImGui::Button(text.c_str())) {
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

        ImVec2 windowPos{pos.x, pos.y + 22};
        ImGui::SetNextWindowPos(windowPos);
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

void AlienImGui::ColorButtonWithPicker(
    std::string const& text,
    uint32_t& color,
    uint32_t& backupColor,
    uint32_t (&savedPalette)[32],
    RealVector2D const& size)
{
    ImVec4 imGuiColor = ImColor(color);
    ImVec4 imGuiBackupColor = ImColor(backupColor);
    ImVec4 imGuiSavedPalette[32];
    for (int i = 0; i < IM_ARRAYSIZE(imGuiSavedPalette); ++i) {
        imGuiSavedPalette[i] = ImColor(savedPalette[i]);
    }

    bool openColorPicker = ImGui::ColorButton(
        text.c_str(), imGuiColor, ImGuiColorEditFlags_NoBorder, ImVec2(ImGui::GetContentRegionAvail().x / 2, 0));
    if (openColorPicker) {
        ImGui::OpenPopup("colorpicker");
        imGuiBackupColor = imGuiColor;
    }
    if (ImGui::BeginPopup("colorpicker")) {
        ImGui::Text("Please choose a color");
        ImGui::Separator();
        ImGui::ColorPicker4(
            "##picker", (float*)&imGuiColor, ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview);
        ImGui::SameLine();

        ImGui::BeginGroup();  // Lock X position
        ImGui::Text("Current");
        ImGui::ColorButton(
            "##current",
            imGuiColor,
            ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf,
            ImVec2(60, 40));
        ImGui::Text("Previous");
        if (ImGui::ColorButton(
                "##previous",
                imGuiBackupColor,
                ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf,
                ImVec2(60, 40))) {
            imGuiColor = imGuiBackupColor;
        }
        ImGui::Separator();
        ImGui::Text("Palette");
        for (int n = 0; n < IM_ARRAYSIZE(imGuiSavedPalette); n++) {
            ImGui::PushID(n);
            if ((n % 8) != 0)
                ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.y);

            ImGuiColorEditFlags paletteButtonFlags =
                ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip;
            if (ImGui::ColorButton("##palette", imGuiSavedPalette[n], paletteButtonFlags, ImVec2(20, 20)))
                imGuiColor = ImVec4(
                    imGuiSavedPalette[n].x,
                    imGuiSavedPalette[n].y,
                    imGuiSavedPalette[n].z,
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
}
