#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include "StyleRepository.h"

_InspectorWindow::_InspectorWindow(CellOrParticleDescription const& entity, RealVector2D const& initialPos)
    : _entity(entity)
    , _initialPos(initialPos)
{}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    if (!_on) {
        return;
    }
    auto width = StyleRepository::getInstance().scaleContent(250.0f);
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, width}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    if (ImGui::Begin(generateTitle().c_str(), &_on)) {
    }
    ImGui::End();
}

bool _InspectorWindow::isClosed() const
{
    return !_on;
}

CellOrParticleDescription _InspectorWindow::getDescription() const
{
    return _entity;
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
