#include "SelectionWindow.h"

#include <imgui.h>

#include "Base/StringHelper.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "EditorModel.h"

_SelectionWindow::_SelectionWindow(EditorModel const& editorModel)
    : _AlienWindow("Selection", "editor.selection", true), _editorModel(editorModel)
{
}

void _SelectionWindow::processIntern()
{
    auto selection = _editorModel->getSelectionShallowData();
    ImGui::Text("Cells");
    ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(selection.numCells).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::Text("Cells from clusters");
    ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(selection.numClusterCells).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::Text("Energy particles");
    ImGui::PushFont(StyleRepository::getInstance().getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::TextUnformatted(StringHelper::format(selection.numParticles).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
}
