#include "InspectorWindow.h"

#include <sstream>
#include <imgui.h>

#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "EngineInterface/DescriptionHelper.h"
#include "EngineInterface/SimulationController.h"
#include "EngineInterface/GenomeDescriptionConverter.h"
#include "EngineInterface/PreviewDescriptionConverter.h"

#include "StyleRepository.h"
#include "Viewport.h"
#include "EditorModel.h"
#include "AlienImGui.h"
#include "CellFunctionStrings.h"
#include "GenomeEditorWindow.h"

using namespace std::string_literals;

namespace
{
    auto const CellWindowWidth = 350.0f;
    auto const ParticleWindowWidth = 280.0f;
    auto const BaseTabTextWidth = 150.0f;
    auto const CellFunctionTextWidth = 180.0f;
    auto const CellFunctionDefenderWidth = 100.0f;
    auto const CellFunctionBaseTabTextWidth = 120.0f;
    auto const ActivityTextWidth = 100.0f;
    auto const GenomeTabTextWidth = 140.0f;
    auto const CellMetadataContentTextWidth = 80.0f;
    auto const ParticleContentTextWidth = 80.0f;

    auto const TreeNodeFlags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;
}

_InspectorWindow::_InspectorWindow(
    SimulationController const& simController,
    Viewport const& viewport,
    EditorModel const& editorModel,
    GenomeEditorWindow const& genomeEditorWindow,
    uint64_t entityId,
    RealVector2D const& initialPos,
    bool selectGenomeTab)
    : _entityId(entityId)
    , _initialPos(initialPos)
    , _viewport(viewport)
    , _editorModel(editorModel)
    , _simController(simController)
    , _genomeEditorWindow(genomeEditorWindow)
    , _selectGenomeTab(selectGenomeTab)
{
}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    if (!_on) {
        return;
    }
    auto width = calcWindowWidth();
    auto height = isCell() ? StyleRepository::getInstance().contentScale(370.0f)
                           : StyleRepository::getInstance().contentScale(70.0f);
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
        auto factor = StyleRepository::getInstance().contentScale(1);

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
        processCellBaseTab(cell);
        processCellFunctionTab(cell);
        processCellFunctionPropertiesTab(cell);
        if (cell.getCellFunctionType() == CellFunction_Constructor) {
            processCellGenomeTab(std::get<ConstructorDescription>(*cell.cellFunction));
        }
        if (cell.getCellFunctionType() == CellFunction_Injector) {
            processCellGenomeTab(std::get<InjectorDescription>(*cell.cellFunction));
        }
        processCellMetadataTab(cell);
        validationAndCorrection(cell);

        ImGui::EndTabBar();

        if (cell != origCell) {
            _simController->changeCell(cell);
        }
    }
}

void _InspectorWindow::processCellBaseTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Base", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            if (ImGui::TreeNodeEx("Properties##Base", TreeNodeFlags)) {
                AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Color").textWidth(BaseTabTextWidth), cell.color);
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(BaseTabTextWidth), cell.energy);
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Position X").format("%.2f").textWidth(BaseTabTextWidth), cell.pos.x);
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Position Y").format("%.2f").textWidth(BaseTabTextWidth), cell.pos.y);
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Velocity X").format("%.2f").textWidth(BaseTabTextWidth), cell.vel.x);
                AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Velocity Y").format("%.2f").textWidth(BaseTabTextWidth), cell.vel.y);
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Stiffness").format("%.2f").step(0.05f).textWidth(BaseTabTextWidth), cell.stiffness);
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Max connections").textWidth(BaseTabTextWidth), cell.maxConnections);
                AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Attach to background").textWidth(BaseTabTextWidth), cell.barrier);
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Connections to other cells", TreeNodeFlags)) {
                for (auto const& [index, connection] : cell.connections | boost::adaptors::indexed(0)) {
                    if (ImGui::TreeNodeEx(("Connection [" + std::to_string(index) + "]").c_str(), ImGuiTreeNodeFlags_None)) {
                        AlienImGui::InputFloat(
                            AlienImGui::InputFloatParameters().name("Reference distance").format("%.2f").textWidth(BaseTabTextWidth).readOnly(true),
                            connection.distance);
                        AlienImGui::InputFloat(
                            AlienImGui::InputFloatParameters().name("Reference angle").format("%.2f").textWidth(BaseTabTextWidth).readOnly(true),
                            connection.angleFromPrevious);
                        ImGui::TreePop();
                    }
                }
                ImGui::TreePop();
            }
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCellFunctionTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Function", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            auto const& parameters = _simController->getSimulationParameters();
            if (ImGui::TreeNodeEx("Properties##Function", TreeNodeFlags)) {
                int type = cell.getCellFunctionType();
                if (AlienImGui::CellFunctionCombo(AlienImGui::CellFunctionComboParameters().name("Function").textWidth(CellFunctionBaseTabTextWidth), type)) {
                    switch (type) {
                    case CellFunction_Neuron: {
                        cell.cellFunction = NeuronDescription();
                    } break;
                    case CellFunction_Transmitter: {
                        cell.cellFunction = TransmitterDescription();
                    } break;
                    case CellFunction_Constructor: {
                        cell.cellFunction = ConstructorDescription();
                    } break;
                    case CellFunction_Sensor: {
                        cell.cellFunction = SensorDescription();
                    } break;
                    case CellFunction_Nerve: {
                        cell.cellFunction = NerveDescription();
                    } break;
                    case CellFunction_Attacker: {
                        cell.cellFunction = AttackerDescription();
                    } break;
                    case CellFunction_Injector: {
                        cell.cellFunction = InjectorDescription();
                    } break;
                    case CellFunction_Muscle: {
                        cell.cellFunction = MuscleDescription();
                    } break;
                    case CellFunction_Defender: {
                        cell.cellFunction = DefenderDescription();
                    } break;
                    case CellFunction_Placeholder: {
                        cell.cellFunction = PlaceHolderDescription();
                    } break;
                    case CellFunction_None: {
                        cell.cellFunction.reset();
                    } break;
                    }
                }

                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Age").textWidth(CellFunctionBaseTabTextWidth), cell.age);
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Activation time").textWidth(CellFunctionBaseTabTextWidth), cell.activationTime);
                AlienImGui::InputInt(
                    AlienImGui::InputIntParameters().name("Execution order").textWidth(CellFunctionBaseTabTextWidth), cell.executionOrderNumber);
                AlienImGui::InputOptionalInt(
                    AlienImGui::InputIntParameters().name("Input").textWidth(CellFunctionBaseTabTextWidth), cell.inputExecutionOrderNumber);
                AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Block Output").textWidth(CellFunctionBaseTabTextWidth), cell.outputBlocked);
                AlienImGui::Combo(
                    AlienImGui::ComboParameters()
                        .name("Living state")
                        .textWidth(CellFunctionBaseTabTextWidth)
                        .values({"Ready", "Under construction", "Just ready", "Dying"}),
                    cell.livingState);
                ImGui::TreePop();
            }
        }
        if (ImGui::TreeNodeEx("Neural activity", TreeNodeFlags)) {
            int index = 0;
            for (auto& channel : cell.activity.channels) {
                AlienImGui::InputFloat(
                    AlienImGui::InputFloatParameters().name("Channel #" + std::to_string(index)).format("%.2f").step(0.1f).textWidth(ActivityTextWidth),
                    channel);
                ++index;
            }
            ImGui::TreePop();
        }

        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCellFunctionPropertiesTab(CellDescription& cell)
{
    if (cell.getCellFunctionType() == CellFunction_None) {
        return;
    }

    std::string title = Const::CellFunctionToStringMap.at(cell.getCellFunctionType());
    if (ImGui::BeginTabItem(title.c_str(), nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            switch (cell.getCellFunctionType()) {
            case CellFunction_Neuron: {
                processNeuronContent(std::get<NeuronDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Transmitter: {
                processTransmitterContent(std::get<TransmitterDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Constructor: {
                processConstructorContent(std::get<ConstructorDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Sensor: {
                processSensorContent(std::get<SensorDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Nerve: {
                processNerveContent(std::get<NerveDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Attacker: {
                processAttackerContent(std::get<AttackerDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Injector: {
                processInjectorContent(std::get<InjectorDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Muscle: {
                processMuscleContent(std::get<MuscleDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Defender: {
                processDefenderContent(std::get<DefenderDescription>(*cell.cellFunction));
            } break;
            case CellFunction_Placeholder: {
            } break;
            }
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

template <typename Description>
void _InspectorWindow::processCellGenomeTab(Description& desc)
{
    auto const& parameters = _simController->getSimulationParameters();

    int flags = ImGuiTabItemFlags_None;
    if (_selectGenomeTab) {
        flags = flags | ImGuiTabItemFlags_SetSelected;
        _selectGenomeTab = false;
    }
    if (ImGui::BeginTabItem("Genome", nullptr, flags)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            AlienImGui::Group("Genome: " + std::to_string(desc.genome.size()) + " bytes");
            if (AlienImGui::Button("Edit")) {
                _genomeEditorWindow->openTab(GenomeDescriptionConverter::convertBytesToDescription(desc.genome));
            }

            ImGui::SameLine();
            if (AlienImGui::Button(AlienImGui::ButtonParameters().buttonText("Retrieve from editor").textWidth(GenomeTabTextWidth))) {
                desc.genome = GenomeDescriptionConverter::convertDescriptionToBytes(_genomeEditorWindow->getCurrentGenome());
                if constexpr (std::is_same<Description, ConstructorDescription>()) {
                    desc.currentGenomePos = 0;
                }
            }

            if constexpr (std::is_same<Description, ConstructorDescription>()) {
                auto entry = GenomeDescriptionConverter::convertByteIndexToCellIndex(desc.genome, desc.currentGenomePos);
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Sequence number").textWidth(GenomeTabTextWidth), entry);
                desc.currentGenomePos = GenomeDescriptionConverter::convertCellIndexToByteIndex(desc.genome, entry);
            }
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Generation").textWidth(GenomeTabTextWidth), desc.genomeGeneration);

            AlienImGui::Group("Preview (approximation)");
            if (ImGui::BeginChild("##child", ImVec2(0, 0), true, ImGuiWindowFlags_HorizontalScrollbar)) {
                auto genomDesc = GenomeDescriptionConverter::convertBytesToDescription(desc.genome);
                auto previewDesc = PreviewDescriptionConverter::convert(genomDesc, std::nullopt, parameters);
                std::optional<int> selectedNodeDummy;
                AlienImGui::ShowPreviewDescription(previewDesc, _genomeZoom, selectedNodeDummy);
            }
            ImGui::EndChild();
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCellMetadataTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Metadata", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            AlienImGui::InputText(AlienImGui::InputTextParameters().name("Name").textWidth(CellMetadataContentTextWidth), cell.metadata.name);

            AlienImGui::InputTextMultiline(
                AlienImGui::InputTextMultilineParameters().name("Notes").textWidth(CellMetadataContentTextWidth).height(100), cell.metadata.description);
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processNerveContent(NerveDescription& nerve)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {

        bool pulseGeneration = nerve.pulseMode > 0;
        if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Generate pulses").textWidth(CellFunctionTextWidth), pulseGeneration)) {
            nerve.pulseMode = pulseGeneration ? 1 : 0;
        }
        if (pulseGeneration) {
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Pulse interval").textWidth(CellFunctionTextWidth), nerve.pulseMode);
            bool alternation = nerve.alternationMode > 0;
            if (AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Alternating pulses").textWidth(CellFunctionTextWidth), alternation)) {
                nerve.alternationMode = alternation ? 1 : 0;
            }
            if (alternation) {
                AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Pulses per phase").textWidth(CellFunctionTextWidth), nerve.alternationMode);
            }
        }
        ImGui::TreePop();
    }
}

void _InspectorWindow::processNeuronContent(NeuronDescription& neuron)
{
    if (ImGui::TreeNodeEx("Neural network", TreeNodeFlags)) {
        AlienImGui::NeuronSelection(
            AlienImGui::NeuronSelectionParameters().outputButtonPositionFromRight(ActivityTextWidth),
            neuron.weights,
            neuron.biases,
            _selectedInput,
            _selectedOutput);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Weight").step(0.05f).textWidth(ActivityTextWidth), neuron.weights[_selectedOutput][_selectedInput]);
        AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Bias").step(0.05f).textWidth(ActivityTextWidth), neuron.biases[_selectedOutput]);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processConstructorContent(ConstructorDescription& constructor)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {
        auto parameters = _simController->getSimulationParameters();

        AlienImGui::Checkbox(AlienImGui::CheckboxParameters().name("Single construction").textWidth(CellFunctionTextWidth), constructor.singleConstruction);
        AlienImGui::Checkbox(
            AlienImGui::CheckboxParameters().name("Separate construction").textWidth(CellFunctionTextWidth), constructor.separateConstruction);
        AlienImGui::InputOptionalInt(
            AlienImGui::InputIntParameters().name("Max connections").textWidth(CellFunctionTextWidth), constructor.maxConnections);
        int constructorMode = constructor.activationMode == 0 ? 0 : 1;
        if (AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Activation mode").textWidth(CellFunctionTextWidth).values({"Manual", "Automatic"}), constructorMode)) {
            constructor.activationMode = constructorMode;
        }
        if (constructorMode == 1) {
            AlienImGui::InputInt(AlienImGui::InputIntParameters().name("Interval").textWidth(CellFunctionTextWidth), constructor.activationMode);
        }
        AlienImGui::AngleAlignmentCombo(
            AlienImGui::AngleAlignmentComboParameters().name("Angle alignment").textWidth(CellFunctionTextWidth), constructor.angleAlignment);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Offspring stiffness").format("%.2f").step(0.1f).textWidth(CellFunctionTextWidth), constructor.stiffness);
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters().name("Offspring activation time").textWidth(CellFunctionTextWidth), constructor.constructionActivationTime);
        ImGui::TreePop();

    }
}

void _InspectorWindow::processInjectorContent(InjectorDescription& injector)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {
        AlienImGui::Combo(
            AlienImGui::ComboParameters().name("Mode").textWidth(CellFunctionTextWidth).values({"Cells under construction", "All Cells"}),
            injector.mode);
        AlienImGui::InputInt(
            AlienImGui::InputIntParameters().name("Counter").textWidth(CellFunctionTextWidth), injector.counter);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processAttackerContent(AttackerDescription& attacker)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {
        AlienImGui::Combo(
            AlienImGui::ComboParameters()
                .name("Energy distribution")
                .values({"Connected cells", "Transmitters and Constructors"})
                .textWidth(CellFunctionTextWidth),
            attacker.mode);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processDefenderContent(DefenderDescription& defender)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {
        AlienImGui::Combo(
            AlienImGui::ComboParameters().name("Mode").values({"Anti-attacker", "Anti-injector"}).textWidth(CellFunctionDefenderWidth),
            defender.mode);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processTransmitterContent(TransmitterDescription& transmitter)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {
        AlienImGui::Combo(
            AlienImGui::ComboParameters()
                .name("Energy distribution")
                .values({"Connected cells", "Transmitters and Constructors"})
                .textWidth(CellFunctionTextWidth),
            transmitter.mode);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processMuscleContent(MuscleDescription& muscle)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {
        AlienImGui::Combo(
            AlienImGui::ComboParameters().name("Mode").values({"Movement", "Contraction and expansion", "Bending"}).textWidth(CellFunctionTextWidth),
            muscle.mode);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processSensorContent(SensorDescription& sensor)
{
    if (ImGui::TreeNodeEx("Properties", TreeNodeFlags)) {
        int mode = sensor.getSensorMode();
        if (AlienImGui::Combo(
                AlienImGui::ComboParameters().name("Mode").values({"Scan vicinity", "Scan specific direction"}).textWidth(CellFunctionTextWidth), mode)) {
            if (mode == SensorMode_Neighborhood) {
                sensor.fixedAngle.reset();
            } else {
                sensor.fixedAngle = 0.0f;
            }
        }
        if (sensor.fixedAngle) {
            AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Scan angle").format("%.1f").textWidth(CellFunctionTextWidth), *sensor.fixedAngle);
        }
        AlienImGui::ComboColor(AlienImGui::ComboColorParameters().name("Scan color").textWidth(CellFunctionTextWidth), sensor.color);
        AlienImGui::InputFloat(
            AlienImGui::InputFloatParameters().name("Min density").format("%.2f").step(0.05f).textWidth(CellFunctionTextWidth), sensor.minDensity);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto origParticle = particle;
    auto energy = toFloat(particle.energy);
    AlienImGui::InputFloat(AlienImGui::InputFloatParameters().name("Energy").textWidth(ParticleContentTextWidth), energy);

    particle.energy = energy;
    if (particle != origParticle) {
        _simController->changeParticle(particle);
    }
}

float _InspectorWindow::calcWindowWidth() const
{
    if (isCell()) {
        return StyleRepository::getInstance().contentScale(CellWindowWidth);
    } else {
        return StyleRepository::getInstance().contentScale(ParticleWindowWidth);
    }
}

void _InspectorWindow::validationAndCorrection(CellDescription& cell) const
{
    auto const& parameters = _simController->getSimulationParameters();

    cell.maxConnections = (cell.maxConnections + MAX_CELL_BONDS + 1) % (MAX_CELL_BONDS + 1);
    cell.executionOrderNumber = (cell.executionOrderNumber + parameters.cellNumExecutionOrderNumbers) % parameters.cellNumExecutionOrderNumbers;
    if (cell.inputExecutionOrderNumber) {
        cell.inputExecutionOrderNumber = (*cell.inputExecutionOrderNumber + parameters.cellNumExecutionOrderNumbers) % parameters.cellNumExecutionOrderNumbers;
    }
    cell.stiffness = std::max(0.0f, std::min(1.0f, cell.stiffness));
    cell.energy = std::max(0.0f, cell.energy);
    switch (cell.getCellFunctionType()) {
    case CellFunction_Constructor: {
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
        if (constructor.maxConnections) {
            constructor.maxConnections = (*constructor.maxConnections + MAX_CELL_BONDS + 1) % (MAX_CELL_BONDS + 1);
        }
        constructor.stiffness = std::max(0.0f, std::min(1.0f, constructor.stiffness));
        constructor.genomeGeneration = std::max(0, constructor.genomeGeneration);
    } break;
    case CellFunction_Sensor: {
        auto& sensor = std::get<SensorDescription>(*cell.cellFunction);
        sensor.minDensity = std::max(0.0f, std::min(1.0f, sensor.minDensity));
    } break;
    case CellFunction_Nerve: {
        auto& nerve = std::get<NerveDescription>(*cell.cellFunction);
        nerve.pulseMode = std::max(0, nerve.pulseMode);
        nerve.alternationMode = std::max(0, nerve.alternationMode);
    } break;
    }
}
