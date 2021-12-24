#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include "EngineInterface/DescriptionHelper.h"
#include "StyleRepository.h"
#include "Viewport.h"

_InspectorWindow::_InspectorWindow(
    Viewport const& viewport,
    CellOrParticleDescription const& entity,
    RealVector2D const& initialPos)
    : _entity(entity)
    , _initialPos(initialPos)
    , _viewport(viewport)
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
    auto windowPos = ImGui::GetWindowPos();
    ImGui::End();

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(_entity));
    drawList->AddLine(windowPos, {entityPos.x, entityPos.y}, ImColor::HSV(0, 0, 1, 0.5));
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
