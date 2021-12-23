#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include "StyleRepository.h"
#include "Viewport.h"

_InspectorWindow::_InspectorWindow(CellOrParticleDescription const& entity, Viewport const& viewport)
    : _entity(entity)
    , _viewport(viewport)
{}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({100.0f, 100.0f}, ImGuiCond_FirstUseEver);

    auto viewport = _viewport->mapWorldToViewPosition(getEntityPos());
    ImGui::SetNextWindowPos({viewport.x, viewport.y + 50}, ImGuiCond_FirstUseEver);
    if (ImGui::Begin(generateTitle().c_str(), &_on)) {
    }
    ImGui::End();
}

bool _InspectorWindow::isClosed() const
{
    return false;
}

std::string _InspectorWindow::generateTitle() const
{
    std::stringstream ss;
    if (std::holds_alternative<CellDescription>(_entity)) {
        ss << "Cell #" << std::hex << std::get<CellDescription>(_entity).id;
    } else {
        ss << "Energy particle #" << std::hex << std::get<ParticleDescription>(_entity).id;
    }
    return ss.str();
}

RealVector2D _InspectorWindow::getEntityPos() const
{
    if (std::holds_alternative<CellDescription>(_entity)) {
        return std::get<CellDescription>(_entity).pos;
    }
    return std::get<ParticleDescription>(_entity).pos;
}
