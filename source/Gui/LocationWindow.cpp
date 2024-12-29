#include "LocationWindow.h"

#include <imgui.h>

#include "StyleRepository.h"

void LocationWindow::init(LocationWidgets const& widgets, RealVector2D const& initialPos)
{
    _widgets = widgets;

    static int id = 0;
    _id = ++id;
    _on = true;
    _initialPos = initialPos;
}

void LocationWindow::process()
{
    ImGui::PushID(_id);

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({scale(650.0f), scale(350.0f)}, ImGuiCond_Once);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Once);
    auto title = _widgets->getLocationName();
    if (ImGui::Begin((title + "###" + std::to_string(_id)).c_str(), &_on)) {
        _widgets->process();
    }
    ImGui::End();

    ImGui::PopID();
}

bool LocationWindow::isOn() const
{
    return _on;
}

int LocationWindow::getLocationIndex() const
{
    return _widgets->getLocationIndex();
}

void LocationWindow::setLocationIndex(int locationIndex)
{
    _widgets->setLocationIndex(locationIndex);
}
