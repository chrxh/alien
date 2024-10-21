#include "ChangeColorDialog.h"

#include <imgui.h>

#include "Fonts/IconsFontAwesome5.h"

#include "EngineInterface/GenomeDescriptionService.h"

#include "AlienImGui.h"
#include "StyleRepository.h"

void ChangeColorDialog::initIntern(std::function<GenomeDescription()> getGenomeFunc, std::function<void(GenomeDescription const&)> setGenomeFunc)
{
    _getGenomeFunc = getGenomeFunc;
    _setGenomeFunc = setGenomeFunc;
}

ChangeColorDialog::ChangeColorDialog()
    : AlienDialog("Change color")
{}

void ChangeColorDialog::processIntern()
{
    AlienImGui::Group("Color transition rule");
    if (ImGui::BeginTable("##", 3, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, scale(20));
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0);
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::PushID("##1");
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().textWidth(0).width(0), _sourceColor);
        ImGui::PopID();

        ImGui::TableSetColumnIndex(1);
        AlienImGui::Text(ICON_FA_LONG_ARROW_ALT_RIGHT);

        ImGui::TableSetColumnIndex(2);
        ImGui::PushID("target color");
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().textWidth(0).width(0), _targetColor);
        ImGui::PopID();

        ImGui::EndTable();
    }
    AlienImGui::Group("Options");
    ImGui::Checkbox("##includeSubgenomes", &_includeSubGenomes);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    AlienImGui::Text("Include sub-genomes");
    
    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienImGui::Separator();

    if (AlienImGui::Button("OK")) {
        auto genome = _getGenomeFunc();
        onChangeColor(genome);
        _setGenomeFunc(genome);
        close();
    }
    ImGui::SetItemDefaultFocus();
    ImGui::SameLine();
    if (AlienImGui::Button("Cancel")) {
        close();
    }
}

void ChangeColorDialog::onChangeColor(GenomeDescription& genome)
{
    for (auto& node : genome.cells) {
        if (node.color == _sourceColor) {
            node.color = _targetColor;
        }
        if (_includeSubGenomes) {
            if (auto subGenome = node.getGenome()) {
                auto subGenomeDesc = GenomeDescriptionService::convertBytesToDescription(*subGenome);
                onChangeColor(subGenomeDesc);
                auto newSubGenome = GenomeDescriptionService::convertDescriptionToBytes(subGenomeDesc);
                node.setGenome(newSubGenome);
            }
        }
    }
}
