#include "SelectionWindow.h"

#include <imgui.h>

#include "Base/StringHelper.h"
#include "StyleRepository.h"
#include "EditorModel.h"

SelectionWindow::SelectionWindow()
    : AlienWindow("Selection", "windows.selection", true)
{}

void SelectionWindow::processIntern()
{
    auto selection = EditorModel::get().getSelectionShallowData();
    ImGui::Text("Cells");
    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(selection.numCells).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::Text("Connected cells");
    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(selection.numClusterCells).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::Text("Energy particles");
    ImGui::PushFont(StyleRepository::get().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(selection.numParticles).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}
