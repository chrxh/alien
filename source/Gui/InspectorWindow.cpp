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

using namespace std::string_literals;

namespace
{
    auto const MaxCellContentTextWidth = 110.0f;
    auto const MaxParticleContentTextWidth = 80.0f;
    auto const CellFunctions =
        std::vector{"Computer"s, "Propulsion"s, "Scanner"s, "Digestion"s, "Constructor"s, "Sensor"s, "Muscle"s};
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
    _cellInstructionMemoryEdit = boost::make_shared<MemoryEditor>();
    _cellInstructionMemoryEdit->OptShowOptions = false;
    _cellInstructionMemoryEdit->OptShowAscii = false;
    _cellInstructionMemoryEdit->OptMidColsCount = 0;
    _cellInstructionMemoryEdit->Cols = 8;

    _cellDataMemoryEdit = boost::make_shared<MemoryEditor>();
    _cellDataMemoryEdit->OptShowOptions = false;
    _cellDataMemoryEdit->OptShowAscii = false;
    _cellDataMemoryEdit->OptMidColsCount = 0;
    _cellDataMemoryEdit->Cols = 8;
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
        auto parameters = _simController->getSimulationParameters();
        int type = static_cast<int>(cell.cellFeature.getType());
        AlienImGui::Combo(
            AlienImGui::ComboParameters()
                              .name("Specialization")
                              .values(CellFunctions)
                              .textWidth(MaxCellContentTextWidth), type);

        auto energy = toFloat(cell.energy);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(MaxCellContentTextWidth), energy);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Max connections")
                .textWidth(MaxCellContentTextWidth)
                .max(parameters.cellMaxBonds)
                .min(0),
            cell.maxConnections);
        AlienImGui::SliderInt(
            AlienImGui::SliderIntParameters()
                .name("Branch number")
                .textWidth(MaxCellContentTextWidth)
                .max(parameters.cellMaxTokenBranchNumber - 1)
                .min(0),
            cell.tokenBranchNumber);
        AlienImGui::Checkbox(
            AlienImGui::CheckBoxParameters().name("Block token").textWidth(MaxCellContentTextWidth), cell.tokenBlocked);

        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCodeTab(CellDescription& cell)
{
    ImGuiTabItemFlags flags = 0;
    if (ImGui::BeginTabItem("Code", nullptr, flags)) {
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
        auto parameters = _simController->getSimulationParameters();
        if (ImGui::BeginChild(
                "##1", ImVec2(0, ImGui::GetContentRegionAvail().y - StyleRepository::getInstance().scaleContent(90)),
                false,
                0)) {
            AlienImGui::Group("Instruction section");
            ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
            auto dataSize = cell.cellFeature.constData.size();
            cell.cellFeature.constData.copy(_cellMemory, dataSize);
            auto maxBytes = parameters.cellFunctionComputerMaxInstructions * 3;
            for (int i = dataSize; i < maxBytes; ++i) {
                _cellMemory[i] = 0;
            }
            _cellDataMemoryEdit->DrawContents(reinterpret_cast<void*>(_cellMemory), maxBytes);
            ImGui::PopFont();
        }
        ImGui::EndChild();
        if (ImGui::BeginChild(
                "##2", ImVec2(0, 0), false, 0)) {
            AlienImGui::Group("Data section");
            ImGui::PushFont(StyleRepository::getInstance().getMonospaceFont());
            auto dataSize = cell.cellFeature.volatileData.size();
            cell.cellFeature.volatileData.copy(_cellMemory, dataSize);
            _cellInstructionMemoryEdit->DrawContents(reinterpret_cast<void*>(_cellMemory), dataSize);
//            std::string t(_cellMemory, dataSize);
            ImGui::PopFont();
            ImGui::EndTabItem();
        }
        ImGui::EndChild();
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto energy = toFloat(particle.energy);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters()
            .name("Energy")
            .textWidth(MaxParticleContentTextWidth),
        energy);
}

