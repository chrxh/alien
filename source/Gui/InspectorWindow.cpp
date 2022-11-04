#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include <boost/algorithm/string.hpp>

#include "ImguiMemoryEditor/imgui_memory_editor.h"
#include "Fonts/IconsFontAwesome5.h"

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "GenomeEditorWindow.h"

using namespace std::string_literals;

namespace
{
    auto const MaxCellContentTextWidth = 150.0f;
    auto const MaxParticleContentTextWidth = 80.0f;
}

_InspectorWindow::_InspectorWindow(
    SimulationController const& simController,
    Viewport const& viewport,
    EditorModel const& editorModel,
    GenomeEditorWindow const& genomeEditorWindow,
    uint64_t entityId,
    RealVector2D const& initialPos)
    : _entityId(entityId)
    , _initialPos(initialPos)
    , _viewport(viewport)
    , _editorModel(editorModel)
    , _simController(simController)
    , _genomeEditorWindow(genomeEditorWindow)
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

void _InspectorWindow::processCell(CellDescription cell)
{
    if (ImGui::BeginTabBar(
            "##CellInspect", /*ImGuiTabBarFlags_AutoSelectNewTabs | */ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        auto origCell = cell;
        showCellGeneralTab(cell);
        showCellInOutChannelTab(cell);
        ImGui::EndTabBar();

        if (cell != origCell) {
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
        if (AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Specialization").values(Const::CellFunctionStrings).textWidth(MaxCellContentTextWidth), type)) {
            switch (type) {
            case Enums::CellFunction_Neuron: {
                cell.cellFunction = NeuronDescription();
            } break;
            case Enums::CellFunction_Transmitter: {
                cell.cellFunction = TransmitterDescription();
            } break;
            case Enums::CellFunction_Constructor: {
                cell.cellFunction = ConstructorDescription();
            } break;
            case Enums::CellFunction_Sensor: {
                cell.cellFunction = SensorDescription();
            } break;
            case Enums::CellFunction_Nerve: {
                cell.cellFunction = NerveDescription();
            } break;
            case Enums::CellFunction_Attacker: {
                cell.cellFunction = AttackerDescription();
            } break;
            case Enums::CellFunction_Injector: {
                cell.cellFunction = InjectorDescription();
            } break;
            case Enums::CellFunction_Muscle: {
                cell.cellFunction = MuscleDescription();
            } break;
            case Enums::CellFunction_Placeholder1: {
                cell.cellFunction = PlaceHolderDescription1();
            } break;
            case Enums::CellFunction_Placeholder2: {
                cell.cellFunction = PlaceHolderDescription2();
            } break;
            case Enums::CellFunction_None: {
                cell.cellFunction.reset();
            } break;
            }
        }

        auto energy = toFloat(cell.energy);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Energy").textWidth(MaxCellContentTextWidth), energy);
        cell.energy = energy;

        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Max connections")
                .textWidth(MaxCellContentTextWidth),
            cell.maxConnections);
        cell.maxConnections = (cell.maxConnections + parameters.cellMaxBonds + 1) % (parameters.cellMaxBonds + 1);
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters()
                .name("Execution order")
                .textWidth(MaxCellContentTextWidth),
            cell.executionOrderNumber);
        cell.executionOrderNumber = (cell.executionOrderNumber + parameters.cellMaxExecutionOrderNumbers) % parameters.cellMaxExecutionOrderNumbers;
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
    if (cell.getCellFunctionType() == Enums::CellFunction_None) {
        return;
    }

    std::string title = Const::CellFunctionToStringMap.at(cell.getCellFunctionType()) + " function";
    if (ImGui::BeginTabItem(title.c_str(), nullptr, ImGuiTabItemFlags_None)) {
        switch (cell.getCellFunctionType()) {
        case Enums::CellFunction_Neuron: {
        } break;
        case Enums::CellFunction_Transmitter: {
        } break;
        case Enums::CellFunction_Constructor: {
            showConstructorContent(std::get<ConstructorDescription>(*cell.cellFunction));
        } break;
        case Enums::CellFunction_Sensor: {
        } break;
        case Enums::CellFunction_Nerve: {
        } break;
        case Enums::CellFunction_Attacker: {
        } break;
        case Enums::CellFunction_Injector: {
        } break;
        case Enums::CellFunction_Muscle: {
        } break;
        case Enums::CellFunction_Placeholder1: {
        } break;
        case Enums::CellFunction_Placeholder2: {
        } break;
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
    if (particle != origParticle) {
        _simController->changeParticle(particle);
    }
}

void _InspectorWindow::showConstructorContent(ConstructorDescription& constructor)
{
    AlienImGui::Group("Properties");
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Single construction").textWidth(MaxCellContentTextWidth), constructor.singleConstruction);
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Separate construction").textWidth(MaxCellContentTextWidth), constructor.separateConstruction);
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Make sticky").textWidth(MaxCellContentTextWidth), constructor.makeSticky);
    int constructorMode = constructor.mode == 0 ? 0 : 1;
    if (AlienImGui::Combo(AlienImGui::ComboParameters().name("Mode").textWidth(MaxCellContentTextWidth).values({"Manual", "Automatic"}), constructorMode)) {
        constructor.mode = constructorMode;
    }
    if (constructorMode == 1) {
        AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Cycles").textWidth(MaxCellContentTextWidth), constructor.mode);
        if (constructor.mode < 0) {
            constructor.mode = 0;
        }
    }

    AlienImGui::Group("Genome");
    auto width = ImGui::GetContentRegionAvail().x;
    if (ImGui::BeginChild("##", ImVec2(width, ImGui::GetTextLineHeight() * 2), true)) {
        AlienImGui::MonospaceText(std::to_string(constructor.genome.size()) + " bytes of genetic information");
    }
    ImGui::EndChild();

    ImGui::BeginDisabled(!_editorModel->getCopiedGenome().has_value());
    if (AlienImGui::Button("Paste")) {
        constructor.genome = *_editorModel->getCopiedGenome();
    }
    ImGui::EndDisabled();

    ImGui::SameLine();
    if (AlienImGui::Button("Edit")) {
        _genomeEditorWindow->openTab(GenomeDescriptionConverter::convertBytesToDescription(constructor.genome, _simController->getSimulationParameters()));
    }

    AlienImGui::Group("Preview");
}

float _InspectorWindow::calcWindowWidth() const
{
    if (isCell()) {
        auto cell = std::get<CellDescription>(_editorModel->getInspectedEntity(_entityId));
        return StyleRepository::getInstance().scaleContent(280.0f);
    }
    return StyleRepository::getInstance().scaleContent(280.0f);
}
