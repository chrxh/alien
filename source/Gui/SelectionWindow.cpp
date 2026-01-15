#include "SelectionWindow.h"

#include <imgui.h>

#include <Base/StringHelper.h>

#include "AlienGui.h"
#include "EditorController.h"
#include "EditorModel.h"
#include "StyleRepository.h"

namespace
{
    auto constexpr ColumnWidth = 150.0f;
}

SelectionWindow::SelectionWindow()
    : AlienWindow("Selection", "windows.selection", true, false, {scale(100), scale(100.0f)})
{}

void SelectionWindow::processIntern()
{
    AlienGui::DynamicTableLayout table(ColumnWidth);
    if (table.begin()) {

        auto selection = EditorModel::get().getSelectionShallowData();
        ImGui::Text("Cells");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        ImGui::TextUnformatted(StringHelper::format(selection.numObjects).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();
        table.next();

        ImGui::Text("Connected cells");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        ImGui::TextUnformatted(StringHelper::format(selection.numClusterCells).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();
        table.next();

        ImGui::Text("Creatures");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        ImGui::TextUnformatted(StringHelper::format(selection.numCreatures).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();
        table.next();

        ImGui::Text("Energy particles");
        ImGui::PushFont(StyleRepository::get().getLargeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor.Value);
        ImGui::TextUnformatted(StringHelper::format(selection.numEnergyParticles).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();
        table.next();

        table.end();
    }
}

bool SelectionWindow::isShown()
{
    return _on && EditorController::get().isOn();
}
