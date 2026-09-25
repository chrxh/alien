#include "LocationWindow.h"

#include <imgui.h>

#include <Data/LocationAccessService.h>

#include <EngineInterface/SimulationFacade.h>

#include "StyleService.h"

void LocationWindow::init(LocationWidget const& widgets, RealVector2D const& initialPos)
{
    _widget = widgets;

    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    _locationId = LocationAccessService::get().getLocationId(parameters, _widget->getOrderNumber());
    _locationType = LocationAccessService::get().getLocationType(_widget->getOrderNumber(), parameters);

    static int id = 0;
    _id = ++id;
    _on = true;
    _initialPos = initialPos;
}

void LocationWindow::process()
{
    auto parameters = _SimulationFacade::get()->getSimulationParameters();
    auto orderNumber = LocationAccessService::get().findOrderNumber(parameters, _locationId);
    if (!orderNumber.has_value() || LocationAccessService::get().getLocationType(orderNumber.value(), parameters) != _locationType) {
        _on = false;
        return;
    }
    _widget->setOrderNumber(orderNumber.value());

    ImGui::PushID(_id);

    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({scale(650.0f), scale(350.0f)}, ImGuiCond_Once);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Once);
    auto title = _widget->getLocationName();
    if (ImGui::Begin((title + "###" + std::to_string(_id)).c_str(), &_on)) {
        _widget->process();
    }
    ImGui::End();

    ImGui::PopID();
}

bool LocationWindow::isOn() const
{
    return _on;
}
