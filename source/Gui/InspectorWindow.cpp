#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include "ImguiMemoryEditor/imgui_memory_editor.h"
#include "EngineInterface/CellComputerCompiler.h"
#include "EngineInterface/DescriptionHelper.h"
#include "EngineImpl/SimulationController.h"
#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"

namespace
{
    auto const MaxParticleContentTextWidth = 100.0f;
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
    _cellMemEdit = boost::make_shared<MemoryEditor>();
    _cellMemEdit->OptShowOptions = false;
    _cellMemEdit->OptShowAscii = false;
    _cellMemEdit->OptMidColsCount = 0;
}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    if (!_on) {
        return;
    }
    auto entity = _editorModel->getInspectedEntity(_entityId);
    auto width = StyleRepository::getInstance().scaleContent(260.0f);
    auto height = isCell() ? StyleRepository::getInstance().scaleContent(280.0f)
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
        processCellGeneralTab(cell);
        if (cell.cellFeature.getType() == Enums::CellFunction::COMPUTER) {
            processCodeTab(cell);
            processMemoryTab(cell);
        }
        ImGui::EndTabBar();
    }
}

void _InspectorWindow::processCellGeneralTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("General", nullptr, ImGuiTabItemFlags_None)) {
        auto energy = toFloat(cell.energy);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters()
                .name("Energy")
                .textWidth(MaxParticleContentTextWidth),
            energy);

        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCodeTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Code", nullptr, ImGuiTabItemFlags_None)) {
        auto sourcecode = CellComputerCompiler::decompileSourceCode(
            cell.cellFeature.constData, _simController->getSymbolMap(), _simController->getSimulationParameters());
        sourcecode.copy(_cellCode, std::min(toInt(sourcecode.length()), IM_ARRAYSIZE(_cellCode) - 1), 0);
        _cellCode[sourcecode.length()] = '\0';
        ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
        ImGui::InputTextMultiline(
            "##source",
            _cellCode,
            IM_ARRAYSIZE(_cellCode),
            ImGui::GetContentRegionAvail(),
            ImGuiInputTextFlags_AllowTabInput);
        ImGui::PopFont();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processMemoryTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Memory", nullptr, ImGuiTabItemFlags_None)) {
        ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
        auto dataSize = cell.cellFeature.volatileData.size();
        cell.cellFeature.volatileData.copy(_cellMemory, dataSize);
        _cellMemEdit->DrawContents(reinterpret_cast<void*>(_cellMemory), dataSize);
        ImGui::PopFont();
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

