#include "SelectionWindow.h"

#include <imgui.h>

#include "Base/StringFormatter.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"
#include "EditorModel.h"

_SelectionWindow::_SelectionWindow(EditorModel const& editorModel)
    : _editorModel(editorModel)
{
    _on = GlobalSettings::getInstance().getBoolState("editor.selection.active", true);
}

_SelectionWindow::~_SelectionWindow()
{
    GlobalSettings::getInstance().setBoolState("editor.selection.active", _on);
}

void _SelectionWindow::process()
{
    if (!_on) {
        return;
    }
    auto selection = _editorModel->getSelectionShallowData();

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    if (ImGui::Begin("Selection", &_on)) {

        ImGui::Text("Cells");
        ImGui::PushFont(StyleRepository::getInstance().getHugeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::TextUnformatted(StringFormatter::format(selection.numCells).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Cells from clusters");
        ImGui::PushFont(StyleRepository::getInstance().getHugeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::TextUnformatted(StringFormatter::format(selection.numClusterCells).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::Text("Energy particles");
        ImGui::PushFont(StyleRepository::getInstance().getHugeFont());
        ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
        ImGui::TextUnformatted(StringFormatter::format(selection.numParticles).c_str());
        ImGui::PopStyleColor();
        ImGui::PopFont();

        ImGui::End();
    }
}

bool _SelectionWindow::isOn() const
{
    return _on;
}

void _SelectionWindow::setOn(bool value)
{
    _on = value;
}