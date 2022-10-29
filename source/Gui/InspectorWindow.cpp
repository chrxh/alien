#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include <boost/algorithm/string.hpp>

#include "ImguiMemoryEditor/imgui_memory_editor.h"
#include "Fonts/IconsFontAwesome5.h"

#include "Base/StringHelper.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"

using namespace std::string_literals;

namespace
{
    auto const MaxCellContentTextWidth = 120.0f;
    auto const MaxParticleContentTextWidth = 80.0f;
    auto const CellFunctionStrings = std::vector{"Neuron"s, "Transmitter"s, "Ribosome"s, "Sensor"s, "Nerve"s, "Attacker"s, "Injector"s, "Muscle"s};
}

_InspectorWindow::_InspectorWindow(
    SimulationController const& simController,
    Viewport const& viewport,
    EditorModel const& editorModel,
    uint64_t entityId,
    RealVector2D const& initialPos)
    : _entityId(entityId)
    , _initialPos(initialPos)
    , _viewport(viewport)
    , _editorModel(editorModel)
    , _simController(simController)
{
}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    if (!_on) {
        return;
    }
    auto width = calcWindowWidth();
    auto height = isCell() ? StyleRepository::getInstance().scaleContent(370.0f)
                           : StyleRepository::getInstance().scaleContent(70.0f);
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    auto entity = _editorModel->getInspectedEntity(_entityId);
    if (ImGui::Begin(generateTitle().c_str(), &_on, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto windowPos = ImGui::GetWindowPos();
        if (isCell()) {
            processCell(std::get<CellDescription>(entity));
        } else {
            processParticle(std::get<ParticleDescription>(entity));
        }
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
    ImGui::End();
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

namespace
{
    bool hasChanges(CellDescription const& left, CellDescription const& right)
    {
        return left.energy != right.energy || left.maxConnections != right.maxConnections || left.underConstruction != right.underConstruction
            || left.inputBlocked != right.inputBlocked || left.outputBlocked != right.outputBlocked
            || left.executionOrderNumber != right.executionOrderNumber
            || left.metadata.name != right.metadata.name || left.metadata.description != right.metadata.description
            || left.barrier != right.barrier;
    }
    bool hasChanges(ParticleDescription const& left, ParticleDescription const& right)
    {
        return left.energy != right.energy;
    }
}

void _InspectorWindow::processCell(CellDescription cell)
{
    if (ImGui::BeginTabBar(
            "##CellInspect", /*ImGuiTabBarFlags_AutoSelectNewTabs | */ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        auto origCell = cell;
        showCellGeneralTab(cell);
        showCellInOutChannelTab(cell);
        ImGui::EndTabBar();

        if (hasChanges(cell, origCell)) {
            _simController->changeCell(cell);
        }
    }
}

void _InspectorWindow::showCellGeneralTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("General", nullptr, ImGuiTabItemFlags_None)) {
        AlienImGui::Group("Properties");
        auto const& parameters = _simController->getSimulationParameters();
        int type = cell.getCellFunctionType();
        AlienImGui::Combo(AlienImGui::ComboParameters().name("Specialization").values(CellFunctionStrings).textWidth(MaxCellContentTextWidth), type);

        auto energy = toFloat(cell.energy);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(MaxCellContentTextWidth), energy);
        cell.energy = energy;

        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Max connections")
                .textWidth(MaxCellContentTextWidth)
                .max(parameters.cellMaxBonds)
                .min(0),
            cell.maxConnections);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Execution order")
                .textWidth(MaxCellContentTextWidth)
                .max(parameters.cellMaxExecutionOrderNumber - 1)
                .min(0),
            cell.executionOrderNumber);
        AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Under construction").textWidth(MaxCellContentTextWidth), cell.underConstruction);
        AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Barrier").textWidth(MaxCellContentTextWidth), cell.barrier);

        AlienImGui::Group("Metadata");

        AlienImGui::InputText(AlienImGui::InputTextParameters().name("Name").textWidth(MaxCellContentTextWidth), cell.metadata.name);

        AlienImGui::InputTextMultiline(
            AlienImGui::InputTextMultilineParameters().name("Notes").textWidth(MaxCellContentTextWidth).height(0), cell.metadata.description);

        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellInOutChannelTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem(ICON_FA_EXCHANGE_ALT " In/out channels", nullptr, ImGuiTabItemFlags_None)) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(Const::InfoTextColor));
        AlienImGui::Text("This is a pure information tab.");
        ImGui::SameLine();
        AlienImGui::HelpMarker("");
        ImGui::PopStyleColor();
        
        if (ImGui::BeginTable(
                "##",
                2,
                ImGuiTableFlags_Resizable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter
                    | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Address", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("Semantic", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            ImGui::TableNextRow();
            switch(cell.getCellFunctionType()){
            case Enums::CellFunction_Ribosome:
                showRibosomeTableContent();
                break;
            case Enums::CellFunction_Attacker:
                showDigestionTableContent();
                break;
            case Enums::CellFunction_Injector:
            case Enums::CellFunction_Nerve:
            case Enums::CellFunction_Neuron:
                showNeuralNetTableContent();
                break;
            case Enums::CellFunction_Sensor:
                showSensorTableContent();
                break;
            case Enums::CellFunction_Transmitter:
            case Enums::CellFunction_Muscle:
                showMuscleTableContent();
                break;
            }
            ImGui::EndTable();
        }

        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto origParticle = particle;
    auto energy = toFloat(particle.energy);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters()
            .name("Energy")
            .textWidth(MaxParticleContentTextWidth),
        energy);

    particle.energy = energy;
    if (hasChanges(particle, origParticle)) {
        _simController->changeParticle(particle);
    }
}

namespace
{
    std::string formatHex(int value)
    {
        std::stringstream stream;
        stream << "0x" << std::hex << static_cast<int>(value);
        return stream.str();
    }
}

void _InspectorWindow::showNeuralNetTableContent()
{
    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Not yet implemented");
}

void _InspectorWindow::showDigestionTableContent()
{
    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);
    AlienImGui::Text("Input: target color (number from 0-6)");
}

void _InspectorWindow::showRibosomeTableContent()
{
    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);
}

void _InspectorWindow::showMuscleTableContent()
{
    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);
}

void _InspectorWindow::showSensorTableContent()
{
    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);

    ImGui::Spacing();
    ImGui::TableNextRow();

    ImGui::TableSetColumnIndex(0);

    ImGui::TableSetColumnIndex(1);
}

float _InspectorWindow::calcWindowWidth() const
{
    if (isCell()) {
        auto cell = std::get<CellDescription>(_editorModel->getInspectedEntity(_entityId));
        return StyleRepository::getInstance().scaleContent(280.0f);
    }
    return StyleRepository::getInstance().scaleContent(280.0f);
}
