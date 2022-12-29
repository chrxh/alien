#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"

#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "GenomeEditorWindow.h"
#include "PreviewDescriptionConverter.h"

using namespace std::string_literals;

namespace
{
    auto const CellWindowWidth = 350.0f;
    auto const ParticleWindowWidth = 280.0f;
    auto const MaxCellContentTextWidth = 180.0f;
    auto const MaxCellMetadataContentTextWidth = 80.0f;
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
        showCellPhysicsTab(cell);
        showCellFunctionTab(cell);
        showCellFunctionPropertiesTab(cell);
        showCellGenomeTab(cell);
        showCellMetadataTab(cell);
        validationAndCorrection(cell);

        ImGui::EndTabBar();

        if (cell != origCell) {
            _simController->changeCell(cell);
        }
    }
}

void _InspectorWindow::showCellPhysicsTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Physics", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(MaxCellContentTextWidth), cell.energy);
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Position X").format("%.2f").textWidth(MaxCellContentTextWidth), cell.pos.x);
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Position Y").format("%.2f").textWidth(MaxCellContentTextWidth), cell.pos.y);
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Velocity X").format("%.2f").textWidth(MaxCellContentTextWidth), cell.vel.x);
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Velocity Y").format("%.2f").textWidth(MaxCellContentTextWidth), cell.vel.y);
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Stiffness").format("%.2f").step(0.05f).textWidth(MaxCellContentTextWidth), cell.stiffness);
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Max connections").textWidth(MaxCellContentTextWidth), cell.maxConnections);
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Attach to background").textWidth(MaxCellContentTextWidth), cell.barrier);
            AlienImGui::Group("Connections to other cells");
            for (auto const& [index, connection] : cell.connections | boost::adaptors::indexed(0)) {
                if (ImGui::TreeNodeEx(("Connection [" + std::to_string(index) + "]").c_str(), ImGuiTreeNodeFlags_None)) {
                    AlienImGui::InputFloat(
                        AlienImGui::InputFloatParameters().name("Reference distance").format("%.2f").textWidth(MaxCellContentTextWidth).readOnly(true),
                        connection.distance);
                    AlienImGui::InputFloat(
                        AlienImGui::InputFloatParameters().name("Reference angle").format("%.2f").textWidth(MaxCellContentTextWidth).readOnly(true),
                        connection.angleFromPrevious);
                    ImGui::TreePop();
                }
            }
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellFunctionTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Function", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            auto const& parameters = _simController->getSimulationParameters();
            int type = cell.getCellFunctionType();
            if (AlienImGui::CellFunctionCombo(AlienImGui::CellFunctionComboParameters().name("Function").textWidth(MaxCellContentTextWidth), type)) {
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

            AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(MaxCellContentTextWidth), cell.color);
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Age").textWidth(MaxCellContentTextWidth), cell.age);
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Activation time").textWidth(MaxCellContentTextWidth), cell.activationTime);
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Execution order").textWidth(MaxCellContentTextWidth), cell.executionOrderNumber);
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block input").textWidth(MaxCellContentTextWidth), cell.inputBlocked);
            AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block Output").textWidth(MaxCellContentTextWidth), cell.outputBlocked);
            AlienImGui::Combo(
                AlienImGui::ComboParameters()
                    .name("Living state")
                    .textWidth(MaxCellContentTextWidth)
                    .values({"Ready", "Under construction", "Just ready", "Dying"}),
                cell.livingState);

        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellFunctionPropertiesTab(CellDescription& cell)
{
    if (cell.getCellFunctionType() == Enums::CellFunction_None) {
        return;
    }

    std::string title = Const::CellFunctionToStringMap.at(cell.getCellFunctionType());
    if (ImGui::BeginTabItem(title.c_str(), nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            switch (cell.getCellFunctionType()) {
            case Enums::CellFunction_Neuron: {
                showNeuronContent(std::get<NeuronDescription>(*cell.cellFunction));
            } break;
            case Enums::CellFunction_Transmitter: {
                showTransmitterContent(std::get<TransmitterDescription>(*cell.cellFunction));
            } break;
            case Enums::CellFunction_Constructor: {
                showConstructorContent(std::get<ConstructorDescription>(*cell.cellFunction));
            } break;
            case Enums::CellFunction_Sensor: {
                showSensorContent(std::get<SensorDescription>(*cell.cellFunction));
            } break;
            case Enums::CellFunction_Nerve: {
                showNerveContent(std::get<NerveDescription>(*cell.cellFunction));
            } break;
            case Enums::CellFunction_Attacker: {
                showAttackerContent(std::get<AttackerDescription>(*cell.cellFunction));
            } break;
            case Enums::CellFunction_Injector: {
            } break;
            case Enums::CellFunction_Muscle: {
                showMuscleContent(std::get<MuscleDescription>(*cell.cellFunction));
            } break;
            case Enums::CellFunction_Placeholder1: {
            } break;
            case Enums::CellFunction_Placeholder2: {
            } break;
            }
            showActivityContent(cell);
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellGenomeTab(CellDescription& cell)
{
    if (cell.getCellFunctionType() != Enums::CellFunction_Constructor) {
        return;
    }
    auto& constructor = std::get<ConstructorDescription>(*cell.cellFunction);
    auto const& parameters = _simController->getSimulationParameters();

    if (ImGui::BeginTabItem("Genome", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
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
                _genomeEditorWindow->openTab(GenomeDescriptionConverter::convertBytesToDescription(constructor.genome, parameters));
            }

            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Read position").textWidth(MaxCellContentTextWidth), constructor.currentGenomePos);

            AlienImGui::Group("Preview");
            if (ImGui::BeginChild("##child", ImVec2(0, StyleRepository::getInstance().scaleContent(200)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
                auto genomDesc = GenomeDescriptionConverter::convertBytesToDescription(constructor.genome, parameters);
                auto previewDesc = PreviewDescriptionConverter::convert(genomDesc, std::nullopt, parameters);
                AlienImGui::ShowPreviewDescription(previewDesc);
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showCellMetadataTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Metadata", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            AlienImGui::InputText(AlienImGui::InputTextParameters().name("Name").textWidth(MaxCellMetadataContentTextWidth), cell.metadata.name);

            AlienImGui::InputTextMultiline(
                AlienImGui::InputTextMultilineParameters().name("Notes").textWidth(MaxCellMetadataContentTextWidth).height(100), cell.metadata.description);
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::showNerveContent(NerveDescription& nerve)
{
    bool pulseGeneration = nerve.pulseMode > 0;
    if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Pulse generation").textWidth(MaxCellContentTextWidth), pulseGeneration)) {
        nerve.pulseMode = pulseGeneration ? 1 : 0; 
    }
    if (pulseGeneration) {
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters().name("Periodicity").textWidth(MaxCellContentTextWidth), nerve.pulseMode);
        bool alternation = nerve.alternationMode > 0;
        if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Alternating pulses").textWidth(MaxCellContentTextWidth), alternation)) {
            nerve.alternationMode = alternation ? 1 : 0;
        }
        if (alternation) {
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Number of pulses").textWidth(MaxCellContentTextWidth), nerve.alternationMode);
        }
    }
}

void _InspectorWindow::showNeuronContent(NeuronDescription& neuron)
{
    AlienImGui::InputFloatMatrix(AlienImGui::InputFloatMatrixParameters().step(0.1f).name("Weights").textWidth(MaxCellMetadataContentTextWidth), neuron.weights);
    AlienImGui::InputFloatVector(AlienImGui::InputFloatVectorParameters().step(0.1f).name("Bias").textWidth(MaxCellMetadataContentTextWidth), neuron.bias);
}

void _InspectorWindow::showConstructorContent(ConstructorDescription& constructor)
{
    auto parameters = _simController->getSimulationParameters();
    
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Single construction").textWidth(MaxCellContentTextWidth), constructor.singleConstruction);
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Separate construction").textWidth(MaxCellContentTextWidth), constructor.separateConstruction);
    AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Adapt max connections").textWidth(MaxCellContentTextWidth), constructor.adaptMaxConnections);
    int constructorMode = constructor.activationMode == 0 ? 0 : 1;
    if (AlienImGui::Combo(AlienImGui::ComboParameters().name("Activation mode").textWidth(MaxCellContentTextWidth).values({"Manual", "Automatic"}), constructorMode)) {
        constructor.activationMode = constructorMode;
    }
    if (constructorMode == 1) {
        AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Periodicity").textWidth(MaxCellContentTextWidth), constructor.activationMode);
    }
    AlienImGui::AngleAlignmentCombo(
        AlienImGui::AngleAlignmentComboParameters().name("Angle alignment").textWidth(MaxCellContentTextWidth), constructor.angleAlignment);
    AlienImGui::InputFloat(
        AlienImGui::InputFloatParameters().name("Offspring stiffness").format("%.2f").step(0.1f).textWidth(MaxCellContentTextWidth), constructor.stiffness);
    AlienImGui::InputInt(
        AlienImGui::InputIntParameters().name("Offspring activation time").textWidth(MaxCellContentTextWidth), constructor.constructionActivationTime);
}

void _InspectorWindow::showAttackerContent(AttackerDescription& attacker)
{
    AlienImGui::Combo(
        AlienImGui::ComboParameters()
            .name("Energy distribution")
            .values({"Connected cells", "Transmitters and Constructors"})
            .textWidth(MaxCellContentTextWidth),
        attacker.mode);
}

void _InspectorWindow::showTransmitterContent(TransmitterDescription& transmitter)
{
    AlienImGui::Combo(
        AlienImGui::ComboParameters()
            .name("Energy distribution")
            .values({"Connected cells", "Transmitters and Constructors"})
            .textWidth(MaxCellContentTextWidth),
        transmitter.mode);
}

void _InspectorWindow::showMuscleContent(MuscleDescription& muscle)
{
    AlienImGui::Combo(
        AlienImGui::ComboParameters()
            .name("Mode").values({"Movement", "Contraction and expansion", "Bending"})
            .textWidth(MaxCellContentTextWidth),
        muscle.mode);
}

void _InspectorWindow::showSensorContent(SensorDescription& sensor)
{
    int mode = sensor.getSensorMode();
    if (AlienImGui::Combo(AlienImGui::ComboParameters().name("Mode").values({"Scan vicinity", "Scan specific region"}).textWidth(MaxCellContentTextWidth), mode)) {
        if (mode == Enums::SensorMode_Neighborhood) {
            sensor.fixedAngle.reset();
        } else {
            sensor.fixedAngle = 0.0f;
        }
    }
    if (sensor.fixedAngle) {
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Scan angle").format("%.1f").textWidth(MaxCellContentTextWidth), *sensor.fixedAngle);
    }
    AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Scan color").textWidth(MaxCellContentTextWidth), sensor.color);
    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Min density").format("%.2f").step(0.05f).textWidth(MaxCellContentTextWidth), sensor.minDensity);
}

void _InspectorWindow::showActivityContent(CellDescription& cell)
{
    AlienImGui::Group("Activity");
    int index = 0;
    for (auto& channel : cell.activity.channels) {
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Channel [" + std::to_string(index) + "]").format("%.2f").step(0.1f).textWidth(MaxCellContentTextWidth),
            channel);
        ++index;
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto origParticle = particle;
    auto energy = toFloat(particle.energy);
    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").textWidth(MaxParticleContentTextWidth), energy);

    particle.energy = energy;
    if (particle != origParticle) {
        _simController->changeParticle(particle);
    }
}

float _InspectorWindow::calcWindowWidth() const
{
    if (isCell()) {
        return StyleRepository::getInstance().scaleContent(CellWindowWidth);
    } else {
        return StyleRepository::getInstance().scaleContent(ParticleWindowWidth);
    }
}

void _InspectorWindow::validationAndCorrection(CellDescription& cell) const
{
    auto const& parameters = _simController->getSimulationParameters();

    cell.maxConnections = (cell.maxConnections + parameters.cellMaxBonds + 1) % (parameters.cellMaxBonds + 1);
    cell.executionOrderNumber = (cell.executionOrderNumber + parameters.cellMaxExecutionOrderNumbers) % parameters.cellMaxExecutionOrderNumbers;
    cell.stiffness = std::max(0.0f, std::min(1.0f, cell.stiffness));
    cell.energy = std::max(0.0f, cell.energy);
    switch (cell.getCellFunctionType()) {
    case Enums::CellFunction_Constructor: {
        auto& constructor = std::get<ConstructorDescription>(*cell.cellFunction);
        if (constructor.currentGenomePos < 0) {
            constructor.currentGenomePos = 0;
        }
        if (constructor.constructionActivationTime < 0) {
            constructor.constructionActivationTime = 0;
        }
        if (constructor.activationMode < 0) {
            constructor.activationMode = 0;
        }
        constructor.stiffness = std::max(0.0f, std::min(1.0f, constructor.stiffness));
    } break;
    case Enums::CellFunction_Sensor: {
        auto& sensor = std::get<SensorDescription>(*cell.cellFunction);
        sensor.minDensity = std::max(0.0f, std::min(1.0f, sensor.minDensity));
    } break;
    case Enums::CellFunction_Nerve: {
        auto& nerve = std::get<NerveDescription>(*cell.cellFunction);
        nerve.pulseMode = std::max(0, nerve.pulseMode);
        nerve.alternationMode = std::max(0, nerve.alternationMode);
    } break;
    }
}
