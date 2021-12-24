#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include "EngineInterface/DescriptionHelper.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"

_InspectorWindow::_InspectorWindow(
    Viewport const& viewport,
    EditorModel const& editorModel,
    uint64_t entityId,
    RealVector2D const& initialPos)
    : _entityId(entityId)
    , _initialPos(initialPos)
    , _viewport(viewport)
    , _editorModel(editorModel)
{}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    if (!_on) {
        return;
    }
    auto entity = _editorModel->getInspectedEntity(_entityId);
    auto width = StyleRepository::getInstance().scaleContent(250.0f);
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, width}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    if (ImGui::Begin(generateTitle().c_str(), &_on)) {
    }
    auto windowPos = ImGui::GetWindowPos();
    ImGui::End();

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(entity));
    drawList->AddLine(
        {windowPos.x + 10.0f, windowPos.y + 10.0f}, {entityPos.x, entityPos.y}, ImColor::HSV(0, 0, 1, 0.8), 2.0f);
}

bool _InspectorWindow::isClosed() const
{
    return !_on;
}

uint64_t _InspectorWindow::getId() const
{
    return _entityId;
}

std::string _InspectorWindow::generateTitle() const
{
    auto entity = _editorModel->getInspectedEntity(_entityId);
    std::stringstream ss;
    if (std::holds_alternative<CellDescription>(entity)) {
        ss << "Cell #" << std::hex << _entityId;
    } else {
        ss << "Energy particle #" << std::hex << _entityId;
    }
    return ss.str();
}
