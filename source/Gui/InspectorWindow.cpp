#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include "EngineInterface/DescriptionHelper.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"

namespace
{
    auto const MaxParticleContentTextWidth = 100.0f;
}

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
    auto width = StyleRepository::getInstance().scaleContent(260.0f);
    auto height = isCell() ? StyleRepository::getInstance().scaleContent(250.0f)
                           : StyleRepository::getInstance().scaleContent(70.0f);
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    if (ImGui::Begin(generateTitle().c_str(), &_on)) {
        if (isCell()) {
            processCell(std::get<CellDescription>(entity));
        } else {
            processParticle(std::get<ParticleDescription>(entity));
        }
    }
    auto windowPos = ImGui::GetWindowPos();
    auto windowSize = ImGui::GetWindowSize();
    ImGui::End();

    ImDrawList* drawList = ImGui::GetBackgroundDrawList();
    auto entityPos = _viewport->mapWorldToViewPosition(DescriptionHelper::getPos(entity));
    auto factor = StyleRepository::getInstance().scaleContent(1);
    drawList->AddLine(
        {windowPos.x + 15.0f * factor, windowPos.y - 5.0f * factor},
        {entityPos.x, entityPos.y},
        Const::InspectorLineColor,
        1.5f);
    drawList->AddRectFilled(
        {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor},
        {windowPos.x + 25.0f * factor, windowPos.y},
        Const::InspectorRectColor,
        1.0,
        0);
    drawList->AddRect(
        {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor},
        {windowPos.x + 25.0f * factor, windowPos.y},
        Const::InspectorLineColor,
        1.0,
        0,
        2.0f);
}

bool _InspectorWindow::isClosed() const
{
    return !_on;
}

uint64_t _InspectorWindow::getId() const
{
    return _entityId;
}

bool _InspectorWindow::isCell() const
{
    auto entity = _editorModel->getInspectedEntity(_entityId);
    return std::holds_alternative<CellDescription>(entity);
}

std::string _InspectorWindow::generateTitle() const
{
    auto entity = _editorModel->getInspectedEntity(_entityId);
    std::stringstream ss;
    if (isCell()) {
        ss << "Cell #" << std::hex << _entityId;
    } else {
        ss << "Energy particle #" << std::hex << _entityId;
    }
    return ss.str();
}

void _InspectorWindow::processCell(CellDescription cell)
{
    if (ImGui::BeginTabBar(
            "##CellInspect", ImGuiTabBarFlags_AutoSelectNewTabs | ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        processCellPropertyTab(cell);
        if (cell.cellFeature.getType() == Enums::CellFunction::COMPUTER) {
            processCodeTab(cell);
        }
        ImGui::EndTabBar();
    }
}

void _InspectorWindow::processCellPropertyTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Properties", nullptr, ImGuiTabItemFlags_None)) {
        auto energy = toFloat(cell.energy);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Energy")
                .defaultValue(cell.energy)
                .textWidth(MaxParticleContentTextWidth),
            energy);
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCodeTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Code", nullptr, ImGuiTabItemFlags_None)) {
        static char text[1024 * 16] = "Test\n";
        ImGui::InputTextMultiline(
            "##source",
            text,
            IM_ARRAYSIZE(text),
            ImVec2(-FLT_MIN, ImGui::GetTextLineHeight() * 15),
            ImGuiInputTextFlags_AllowTabInput);
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto energy = toFloat(particle.energy);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters()
            .name("Energy")
            .defaultValue(particle.energy)
            .textWidth(MaxParticleContentTextWidth),
        energy);
}

