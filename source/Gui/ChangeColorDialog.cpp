#include "ChangeColorDialog.h"

#include <imgui.h>

#include <Fonts/IconsFontAwesome5.h>

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "StyleRepository.h"

void ChangeColorDialog::open(GenomeTabEditData const& editData)
{
    _editData = editData;
    AlienDialog::open();
}

ChangeColorDialog::ChangeColorDialog()
    : AlienDialog("Change color")
{}

void ChangeColorDialog::processIntern()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Color transition rule"));
    if (ImGui::BeginTable("##", 3, ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, scale(20));
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthStretch, 0);
        ImGui::TableNextRow();

        ImGui::TableSetColumnIndex(0);
        ImGui::PushID("##1");
        AlienGui::ComboColor(AlienGui::ComboColorParameters().textWidth(0).width(0), _sourceColor);
        ImGui::PopID();

        ImGui::TableSetColumnIndex(1);
        AlienGui::Text(AlienGui::TextParameters().text(ICON_FA_LONG_ARROW_ALT_RIGHT));

        ImGui::TableSetColumnIndex(2);
        ImGui::PushID("target color");
        AlienGui::ComboColor(AlienGui::ComboColorParameters().textWidth(0).width(0), _targetColor);
        ImGui::PopID();

        ImGui::EndTable();
    }
    AlienGui::Group(AlienGui::GroupParameters().text("Options"));
    ImGui::Checkbox("##restrictToSelectedGene", &_restrictToSelectedGene);
    ImGui::SameLine(0, ImGui::GetStyle().FramePadding.x * 4);
    AlienGui::Text(AlienGui::TextParameters().text("Restrict to selected gene"));

    ImGui::Dummy({0, ImGui::GetContentRegionAvail().y - scale(50.0f)});
    AlienGui::Separator();

    if (AlienGui::Button("OK")) {
        onChangeColor();
        close();
    }
    ImGui::SetItemDefaultFocus();
    ImGui::SameLine();
    if (AlienGui::Button("Cancel")) {
        close();
    }
}

void ChangeColorDialog::onChangeColor()
{
    if (_restrictToSelectedGene && _editData->selectedGeneIndex.has_value()) {
        auto& gene = _editData->getSelectedGeneRef();
        for (auto& node : gene._nodes) {
            if (node._color == _sourceColor) {
                node._color = _targetColor;
            }
        }
    }
    if (!_restrictToSelectedGene) {
        for (auto& gene : _editData->genome._genes) {
            for (auto& node : gene._nodes) {
                if (node._color == _sourceColor) {
                    node._color = _targetColor;
                }
            }
        }
    }
}
