#include "SpatialControlWindow.h"

#include "imgui.h"

#include "Base/StringFormatter.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "Viewport.h"

_SpatialControlWindow::_SpatialControlWindow(
    SimulationController const& simController,
    Viewport const& viewport,
    StyleRepository const& styleRepository)
    : _simController(simController)
    , _viewport(viewport)
    , _styleRepository(styleRepository)
{}

void _SpatialControlWindow::process()
{
    if (!_on) {
        return;
    }

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha);
    ImGui::Begin("Spatial control", &_on);

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    ImGui::Spacing();

    ImGui::BeginTable("table1", 3, 0/*ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg*/);

    //world size
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("World size");

    ImGui::TableSetColumnIndex(1);
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    auto worldSize = _simController->getWorldSize();
    ImGui::Text((StringFormatter::format(worldSize.x) + " x " + StringFormatter::format(worldSize.y)).c_str());
    ImGui::PopStyleColor();

    //zoom factor
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("Zoom factor");

    ImGui::TableSetColumnIndex(1);
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    ImGui::Text(StringFormatter::format(_viewport->getZoomFactor(), 1).c_str());
    ImGui::PopStyleColor();

    //view center position
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::Text("View center position");

    ImGui::TableSetColumnIndex(1);
    ImGui::PushStyleColor(ImGuiCol_Text, Const::TextDecentColor);
    auto centerPos = _viewport->getCenterInWorldPos();
    ImGui::Text((StringFormatter::format(centerPos.x, 1) + " x " + StringFormatter::format(centerPos.y, 1)).c_str());
    ImGui::PopStyleColor();

    ImGui::EndTable();

    ImGui::End();
}

bool _SpatialControlWindow::isOn() const
{
    return _on;
}

void _SpatialControlWindow::setOn(bool value)
{
    _on = value;
}
