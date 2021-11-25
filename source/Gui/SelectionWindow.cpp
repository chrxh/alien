#include "SelectionWindow.h"

#include "imgui.h"

#include "Base/StringFormatter.h"
#include "StyleRepository.h"
#include "GlobalSettings.h"

_SelectionWindow::_SelectionWindow(StyleRepository const& styleRepository)
    : _styleRepository(styleRepository)
{
    _on = GlobalSettings::getInstance().getBoolState("editor.selection.active", false);
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

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::Begin("Selection", &_on);

    ImGui::Text("Cells");
    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::Text(StringFormatter::format(_selection.numCells).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::Text("Connected cells");
    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::Text(StringFormatter::format(_selection.numIndirectCells).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::Text("Energy particles");
    ImGui::PushFont(_styleRepository->getLargeFont());
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::Text(StringFormatter::format(_selection.numParticles).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();

    ImGui::End();
}

bool _SelectionWindow::isOn() const
{
    return _on;
}

void _SelectionWindow::setOn(bool value)
{
    _on = value;
}

void _SelectionWindow::setSelection(SelectedEntities const& selection)
{
    _selection = selection;
}
