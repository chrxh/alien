#include "LocationWindow.h"

#include <imgui.h>

#include "StyleRepository.h"

void LocationWindow::init(std::string const& title, LocationWidgets const& widgets)
{
    _title = title;
    _widgets = widgets;

    static int id = 0;
    _id = ++id;
}

void LocationWindow::process()
{
    ImGui::PushID(_id);

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({scale(650.0f), scale(350.0f)}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin((_title + "##" + std::to_string(_id)).c_str(), &_on)) {
        _widgets->process();
    }
    ImGui::End();

    ImGui::PopID();
}
