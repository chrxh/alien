#include "InspectorWindow.h"

#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include "EngineInterface/DescriptionEditService.h"
#include "EngineInterface/GenomeDescriptionConverterService.h"
#include "EngineInterface/PreviewDescriptionService.h"
#include "EngineInterface/SimulationFacade.h"

#include "AlienGui.h"
#include "EditorModel.h"
#include "GenomeEditorWindow.h"
#include "HelpStrings.h"
#include "OverlayController.h"
#include "StyleRepository.h"
#include "Viewport.h"

using namespace std::string_literals;

namespace
{
    auto const CellWindowWidth = 380.0f;
    auto const ParticleWindowWidth = 280.0f;
    auto const BaseTabTextWidth = 162.0f;
    auto const CellTypeTextWidth = 195.0f;
    auto const CellTypeDefenderWidth = 100.0f;
    auto const CellTypeBaseTabTextWidth = 150.0f;
    auto const SignalTextWidth = 130.0f;
    auto const GenomeTabTextWidth = 195.0f;
    auto const ParticleContentTextWidth = 80.0f;

    auto const TreeNodeFlags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;
}

_InspectorWindow::_InspectorWindow(SimulationFacade const& simulationFacade, uint64_t entityId, RealVector2D const& initialPos, bool selectGenomeTab)
    : _entityId(entityId)
    , _initialPos(initialPos)
    , _simulationFacade(simulationFacade)
    , _selectGenomeTab(selectGenomeTab)
{}

_InspectorWindow::~_InspectorWindow() {}

void _InspectorWindow::process()
{
    if (!_on) {
        return;
    }
    auto width = calcWindowWidth();
    auto height = isCell() ? StyleRepository::get().scale(370.0f) : StyleRepository::get().scale(70.0f);
    auto borderlessRendering = _simulationFacade->getSimulationParameters().borderlessRendering.value;
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    auto entity = EditorModel::get().getInspectedEntity(_entityId);
    if (ImGui::Begin(generateTitle().c_str(), &_on, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto windowPos = ImGui::GetWindowPos();
        if (isCell()) {
            processCell(std::get<CellDescription>(entity));
        } else {
            processParticle(std::get<ParticleDescription>(entity));
        }
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescriptionEditService::get().getPos(entity), borderlessRendering);
        auto factor = StyleRepository::get().scale(1);

        drawList->AddLine({windowPos.x + 15.0f * factor, windowPos.y - 5.0f * factor}, {entityPos.x, entityPos.y}, Const::InspectorLineColor, 1.5f);
        drawList->AddRectFilled(
            {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor}, {windowPos.x + 25.0f * factor, windowPos.y}, Const::InspectorRectColor, 1.0, 0);
        drawList->AddRect(
            {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor}, {windowPos.x + 25.0f * factor, windowPos.y}, Const::InspectorLineColor, 1.0, 0, 2.0f);
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
    auto entity = EditorModel::get().getInspectedEntity(_entityId);
    return std::holds_alternative<CellDescription>(entity);
}

std::string _InspectorWindow::generateTitle() const
{
    auto entity = EditorModel::get().getInspectedEntity(_entityId);
    std::stringstream ss;
    if (isCell()) {
        ss << "Cell with id 0x" << std::hex << std::uppercase << _entityId;
    } else {
        ss << "Energy particle with id 0x" << std::hex << std::uppercase << _entityId;
    }
    return ss.str();
}

void _InspectorWindow::processCell(CellDescription cell)
{
    if (ImGui::BeginTabBar("##CellInspect", /*ImGuiTabBarFlags_AutoSelectNewTabs | */ ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        auto origCell = cell;
        processCellGeneralTab(cell);
        processCellTypeTab(cell);
        processCellTypePropertiesTab(cell);
        if (cell.getCellType() == CellType_Constructor) {
            processCellGenomeTab(std::get<ConstructorDescription>(cell._cellTypeData));
        }
        if (cell.getCellType() == CellType_Injector) {
            processCellGenomeTab(std::get<InjectorDescription>(cell._cellTypeData));
        }
        processCellMetadataTab(cell);
        validateAndCorrect(cell);

        ImGui::EndTabBar();

        if (cell != origCell) {
            _simulationFacade->changeCell(cell);
        }
    }
}

void _InspectorWindow::processCellGeneralTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("General", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            if (ImGui::TreeNodeEx("Properties###general", TreeNodeFlags)) {
                std::stringstream ss;
                ss << "0x" << std::hex << std::uppercase << cell._id;
                auto cellId = ss.str();

                AlienGui::ComboColor(
                    AlienGui::ComboColorParameters().name("Color").textWidth(BaseTabTextWidth).tooltip(Const::GenomeColorTooltip), cell._color);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(BaseTabTextWidth).tooltip(Const::CellEnergyTooltip), cell._energy);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Age").textWidth(BaseTabTextWidth).tooltip(Const::CellAgeTooltip), cell._age);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Position X").format("%.2f").textWidth(BaseTabTextWidth), cell._pos.x);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Position Y").format("%.2f").textWidth(BaseTabTextWidth), cell._pos.y);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Velocity X").format("%.2f").textWidth(BaseTabTextWidth), cell._vel.x);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Velocity Y").format("%.2f").textWidth(BaseTabTextWidth), cell._vel.y);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters()
                        .name("Stiffness")
                        .format("%.2f")
                        .step(0.05f)
                        .textWidth(BaseTabTextWidth)
                        .tooltip(Const::CellStiffnessTooltip),
                    cell._stiffness);
                AlienGui::Checkbox(
                    AlienGui::CheckboxParameters().name("Sticky").textWidth(BaseTabTextWidth).tooltip(Const::CellIndestructibleTooltip), cell._sticky);
                AlienGui::Checkbox(
                    AlienGui::CheckboxParameters().name("Indestructible wall").textWidth(BaseTabTextWidth).tooltip(Const::CellIndestructibleTooltip),
                    cell._barrier);
                AlienGui::InputText(
                    AlienGui::InputTextParameters().name("Cell id").textWidth(BaseTabTextWidth).tooltip(Const::CellIdTooltip).readOnly(true), cellId);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("TEMP: abs angle to conn0").format("%.1f").textWidth(BaseTabTextWidth), cell._angleToFront);
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Signal routing", TreeNodeFlags)) {
                AlienGui::Checkbox(
                    AlienGui::CheckboxParameters().name("Signal routing restriction").textWidth(BaseTabTextWidth), cell._signalRoutingRestriction._active);
                if (cell._signalRoutingRestriction._active) {
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Signal base angle").format("%.1f").step(2.0f).textWidth(BaseTabTextWidth),
                        cell._signalRoutingRestriction._baseAngle);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Signal opening angle").format("%.1f").step(2.0f).textWidth(BaseTabTextWidth),
                        cell._signalRoutingRestriction._openingAngle);
                }
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Associated creature##Base", TreeNodeFlags)) {
                std::stringstream ss;
                ss << "0x" << std::hex << std::uppercase << cell._creatureId;
                auto creatureId = ss.str();
                AlienGui::InputText(
                    AlienGui::InputTextParameters().name("Creature id").textWidth(BaseTabTextWidth).tooltip(Const::CellIdTooltip).readOnly(true), creatureId);
                AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Mutation id").textWidth(BaseTabTextWidth).tooltip(Const::CellMutationIdTooltip), cell._mutationId);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Genome complexity").textWidth(BaseTabTextWidth).tooltip(Const::GenomeComplexityTooltip),
                    cell._genomeComplexity);

                ImGui::TreePop();
            }
            if (ImGui::TreeNodeEx("Connections to other cells", TreeNodeFlags)) {
                for (auto const& [index, connection] : cell._connections | boost::adaptors::indexed(0)) {
                    if (ImGui::TreeNodeEx(("Connection [" + std::to_string(index) + "]").c_str(), ImGuiTreeNodeFlags_None)) {
                        std::stringstream ss;
                        ss << "0x" << std::hex << std::uppercase << connection._cellId;
                        auto cellId = ss.str();

                        AlienGui::InputText(
                            AlienGui::InputTextParameters().name("Cell id").textWidth(BaseTabTextWidth).tooltip(Const::CellIdTooltip).readOnly(true), cellId);
                        AlienGui::InputFloat(
                            AlienGui::InputFloatParameters()
                                .name("Reference distance")
                                .format("%.2f")
                                .textWidth(BaseTabTextWidth)
                                .readOnly(true)
                                .tooltip(Const::CellReferenceDistanceTooltip),
                            connection._distance);
                        AlienGui::InputFloat(
                            AlienGui::InputFloatParameters()
                                .name("Reference angle")
                                .format("%.2f")
                                .textWidth(BaseTabTextWidth)
                                .readOnly(true)
                                .tooltip(Const::CellReferenceAngleTooltip),
                            connection._angleFromPrevious);
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

void _InspectorWindow::processCellTypeTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Function", nullptr, ImGuiTabItemFlags_None)) {
        int type = cell.getCellType();
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

            if (cell._neuralNetwork.has_value()) {
                processNeuronContent(cell);
            }

            if (ImGui::TreeNodeEx("Properties###type", TreeNodeFlags)) {
                if (AlienGui::Combo(
                        AlienGui::ComboParameters()
                            .name("Function")
                            .values(Const::CellTypeStrings)
                            .textWidth(CellTypeBaseTabTextWidth)
                            .tooltip(Const::getCellTypeTooltip(type)),
                        type)) {
                    switch (type) {
                    case CellType_Structure: {
                        cell._cellTypeData = StructureCellDescription();
                    } break;
                    case CellType_Free: {
                        cell._cellTypeData = FreeCellDescription();
                    } break;
                    case CellType_Base: {
                        cell._cellTypeData = BaseDescription();
                    } break;
                    case CellType_Depot: {
                        cell._cellTypeData = DepotDescription();
                    } break;
                    case CellType_Constructor: {
                        cell._cellTypeData = ConstructorDescription();
                    } break;
                    case CellType_Sensor: {
                        cell._cellTypeData = SensorDescription();
                    } break;
                    case CellType_Oscillator: {
                        cell._cellTypeData = OscillatorDescription();
                    } break;
                    case CellType_Attacker: {
                        cell._cellTypeData = AttackerDescription();
                    } break;
                    case CellType_Injector: {
                        cell._cellTypeData = InjectorDescription();
                    } break;
                    case CellType_Muscle: {
                        cell._cellTypeData = MuscleDescription();
                    } break;
                    case CellType_Defender: {
                        cell._cellTypeData = DefenderDescription();
                    } break;
                    case CellType_Reconnector: {
                        cell._cellTypeData = ReconnectorDescription();
                    } break;
                    case CellType_Detonator: {
                        cell._cellTypeData = DetonatorDescription();
                    } break;
                    }
                }

                AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Activation time")
                        .textWidth(CellTypeBaseTabTextWidth)
                        .tooltip(Const::GenomeConstructorOffspringActivationTime),
                    cell._activationTime);
                AlienGui::Combo(
                    AlienGui::ComboParameters()
                        .name("Living state")
                        .textWidth(CellTypeBaseTabTextWidth)
                        .values({"Ready", "Under construction", "Activating", "Detached", "Reviving", "Dying"})
                        .tooltip(Const::CellLivingStateTooltip),
                    cell._livingState);
                ImGui::TreePop();
            }
        }
        if (cell._signal.has_value()) {
            if (ImGui::TreeNodeEx("Signals", TreeNodeFlags)) {
                int index = 0;
                for (auto& channel : cell._signal->_channels) {
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Channel #" + std::to_string(index)).format("%.3f").step(0.1f).textWidth(SignalTextWidth),
                        channel);
                    ++index;
                }
                ImGui::TreePop();
            }
        }

        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCellTypePropertiesTab(CellDescription& cell)
{
    if (cell.getCellType() == CellType_Structure || cell.getCellType() == CellType_Free) {
        return;
    }

    std::string title = Const::CellTypeStrings.at(cell.getCellType());
    if (ImGui::BeginTabItem(title.c_str(), nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            switch (cell.getCellType()) {
            case CellType_Base: {
            } break;
            case CellType_Depot: {
                processTransmitterContent(std::get<DepotDescription>(cell._cellTypeData));
            } break;
            case CellType_Constructor: {
                processConstructorContent(std::get<ConstructorDescription>(cell._cellTypeData));
            } break;
            case CellType_Sensor: {
                processSensorContent(std::get<SensorDescription>(cell._cellTypeData));
            } break;
            case CellType_Oscillator: {
                processOscillatorContent(std::get<OscillatorDescription>(cell._cellTypeData));
            } break;
            case CellType_Attacker: {
                processAttackerContent(std::get<AttackerDescription>(cell._cellTypeData));
            } break;
            case CellType_Injector: {
                processInjectorContent(std::get<InjectorDescription>(cell._cellTypeData));
            } break;
            case CellType_Muscle: {
                processMuscleContent(std::get<MuscleDescription>(cell._cellTypeData));
            } break;
            case CellType_Defender: {
                processDefenderContent(std::get<DefenderDescription>(cell._cellTypeData));
            } break;
            case CellType_Reconnector: {
                processReconnectorContent(std::get<ReconnectorDescription>(cell._cellTypeData));
            } break;
            case CellType_Detonator: {
                processDetonatorContent(std::get<DetonatorDescription>(cell._cellTypeData));
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
    auto const& parameters = _simulationFacade->getSimulationParameters();

    int flags = ImGuiTabItemFlags_None;
    if (_selectGenomeTab) {
        flags = flags | ImGuiTabItemFlags_SetSelected;
        _selectGenomeTab = false;
    }
    if (ImGui::BeginTabItem("Genome", nullptr, flags)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

            auto previewNodeResult = ImGui::TreeNodeEx("Preview (reference configuration)", TreeNodeFlags);
            AlienGui::HelpMarker(Const::GenomePreviewTooltip);
            if (previewNodeResult) {
                if (ImGui::BeginChild("##child", ImVec2(0, scale(200)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
                    auto genomDesc = GenomeDescriptionConverterService::get().convertBytesToDescription(desc._genome);
                    auto previewDesc = PreviewDescriptionService::get().convert(genomDesc, std::nullopt, parameters);
                    std::optional<int> selectedNodeDummy;
                    AlienGui::ShowPreviewDescription(previewDesc, _genomeZoom, selectedNodeDummy);
                }
                ImGui::EndChild();
                if (AlienGui::Button("Edit")) {
                    GenomeEditorWindow::get().openTab(GenomeDescriptionConverterService::get().convertBytesToDescription(desc._genome));
                }

                ImGui::SameLine();
                if (AlienGui::Button(AlienGui::ButtonParameters().buttonText("Inject from editor").textWidth(ImGui::GetContentRegionAvail().x))) {
                    printOverlayMessage("Genome injected");
                    desc._genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeEditorWindow::get().getCurrentGenome());
                    if constexpr (std::is_same<Description, ConstructorDescription>()) {
                        desc._genomeCurrentNodeIndex = 0;
                        desc.numInheritedGenomeNodes(0);
                    }
                }
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Properties (entire genome)", TreeNodeFlags)) {
                auto numNodes = toInt(GenomeDescriptionConverterService::get().getNumNodesRecursively(desc._genome, true));
                AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Number of cells")
                        .textWidth(GenomeTabTextWidth)
                        .readOnly(true)
                        .tooltip(Const::GenomeNumCellsRecursivelyTooltip),
                    numNodes);

                auto numBytes = toInt(desc._genome.size());
                AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Bytes").textWidth(GenomeTabTextWidth).readOnly(true).tooltip(Const::GenomeBytesTooltip), numBytes);

                AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Generation").textWidth(GenomeTabTextWidth).tooltip(Const::GenomeGenerationTooltip),
                    desc._genomeGeneration);
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Properties (principal genome part)", TreeNodeFlags)) {

                auto genomeDesc = GenomeDescriptionConverterService::get().convertBytesToDescription(desc._genome);
                auto numBranches = genomeDesc._header.getNumBranches();
                AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Number of branches")
                        .textWidth(GenomeTabTextWidth)
                        .readOnly(true)
                        .tooltip(Const::GenomeNumBranchesTooltip),
                    numBranches);

                auto numRepetitions = genomeDesc._header._numRepetitions;
                AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Repetitions per branch")
                        .textWidth(GenomeTabTextWidth)
                        .infinity(true)
                        .readOnly(true)
                        .tooltip(Const::GenomeRepetitionsPerBranchTooltip),
                    numRepetitions);

                auto numNodes = toInt(genomeDesc._cells.size());
                AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Cells per repetition")
                        .textWidth(GenomeTabTextWidth)
                        .readOnly(true)
                        .tooltip(Const::GenomeNumCellsTooltip),
                    numNodes);

                if constexpr (std::is_same<Description, ConstructorDescription>()) {
                    AlienGui::InputInt(
                        AlienGui::InputIntParameters().name("Current branch index").textWidth(GenomeTabTextWidth).tooltip(Const::GenomeCurrentBranchTooltip),
                        desc._genomeCurrentBranch);
                    AlienGui::InputInt(
                        AlienGui::InputIntParameters()
                            .name("Current repetition index")
                            .textWidth(GenomeTabTextWidth)
                            .tooltip(Const::GenomeCurrentRepetitionTooltip),
                        desc._genomeCurrentRepetition);
                    AlienGui::InputInt(
                        AlienGui::InputIntParameters().name("Current cell index").textWidth(GenomeTabTextWidth).tooltip(Const::GenomeCurrentCellTooltip),
                        desc._genomeCurrentNodeIndex);
                }
                ImGui::TreePop();
            }
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processCellMetadataTab(CellDescription& cell)
{
    if (ImGui::BeginTabItem("Annotation", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, 0)) {
            AlienGui::InputText(AlienGui::InputTextParameters().hint("Name").textWidth(0), cell._metadata._name);

            AlienGui::InputTextMultiline(AlienGui::InputTextMultilineParameters().hint("Notes").textWidth(0).height(100), cell._metadata._description);
        }
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}

void _InspectorWindow::processOscillatorContent(OscillatorDescription& oscillator)
{
    if (ImGui::TreeNodeEx("Properties###oscillator", TreeNodeFlags)) {

        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Pulse interval").textWidth(CellTypeTextWidth).tooltip(Const::GenomeOscillatorPulseIntervalTooltip),
            oscillator._autoTriggerInterval);
        bool alternation = oscillator._alternationInterval > 0;
        if (AlienGui::Checkbox(
                AlienGui::CheckboxParameters().name("Alternating pulses").textWidth(CellTypeTextWidth).tooltip(Const::GenomeOscillatorAlternatingPulsesTooltip),
                alternation)) {
            oscillator._alternationInterval = alternation ? 1 : 0;
        }
        if (alternation) {
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Pulses per phase").textWidth(CellTypeTextWidth).tooltip(Const::GenomeOscillatorPulsesPerPhaseTooltip),
                oscillator._alternationInterval);
        }
        ImGui::TreePop();
    }
}

void _InspectorWindow::processNeuronContent(CellDescription& cell)
{
    if (ImGui::TreeNodeEx("Neural network", TreeNodeFlags)) {
        AlienGui::NeuronSelection(
            AlienGui::NeuronSelectionParameters().rightMargin(0),
            cell._neuralNetwork->_weights,
            cell._neuralNetwork->_biases,
            cell._neuralNetwork->_activationFunctions);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processConstructorContent(ConstructorDescription& constructor)
{
    if (ImGui::TreeNodeEx("Properties###constructor", TreeNodeFlags)) {
        int constructorMode = constructor._autoTriggerInterval == 0 ? 0 : 1;
        if (AlienGui::Combo(
                AlienGui::ComboParameters()
                    .name("Activation mode")
                    .textWidth(CellTypeTextWidth)
                    .values({"Manual", "Automatic"})
                    .tooltip(Const::GenomeConstructorActivationModeTooltip),
                constructorMode)) {
            constructor._autoTriggerInterval = constructorMode;
        }
        if (constructorMode == 1) {
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Interval").textWidth(CellTypeTextWidth).tooltip(Const::GenomeConstructorIntervalTooltip),
                constructor._autoTriggerInterval);
        }
        AlienGui::InputInt(
            AlienGui::InputIntParameters()
                .name("Offspring activation time")
                .textWidth(CellTypeTextWidth)
                .tooltip(Const::GenomeConstructorOffspringActivationTime),
            constructor._constructionActivationTime);
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Construction angle #1")
                .textWidth(CellTypeTextWidth)
                .format("%.1f")
                .tooltip(Const::GenomeConstructorAngle1Tooltip),
            constructor._constructionAngle1);
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Construction angle #2")
                .textWidth(CellTypeTextWidth)
                .format("%.1f")
                .tooltip(Const::GenomeConstructorAngle2Tooltip),
            constructor._constructionAngle2);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processInjectorContent(InjectorDescription& injector)
{
    if (ImGui::TreeNodeEx("Properties###injector", TreeNodeFlags)) {
        AlienGui::Combo(
            AlienGui::ComboParameters()
                .name("Mode")
                .textWidth(CellTypeTextWidth)
                .values({"Only empty cells", "All cells"})
                .tooltip(Const::GenomeInjectorModeTooltip),
            injector._mode);
        ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Process data", TreeNodeFlags)) {
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Counter").textWidth(CellTypeTextWidth).tooltip(Const::CellInjectorCounterTooltip), injector._counter);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processAttackerContent(AttackerDescription& attacker)
{
    if (ImGui::TreeNodeEx("Properties###attacker", TreeNodeFlags)) {
        ImGui::TreePop();
    }
}

void _InspectorWindow::processDefenderContent(DefenderDescription& defender)
{
    if (ImGui::TreeNodeEx("Properties###defender", TreeNodeFlags)) {
        AlienGui::Combo(
            AlienGui::ComboParameters()
                .name("Mode")
                .values({"Anti-attacker", "Anti-injector"})
                .textWidth(CellTypeDefenderWidth)
                .tooltip(Const::GenomeDefenderModeTooltip),
            defender._mode);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processTransmitterContent(DepotDescription& transmitter)
{
    if (ImGui::TreeNodeEx("Properties###transmitter", TreeNodeFlags)) {
        AlienGui::Combo(
            AlienGui::ComboParameters()
                .name("Energy distribution")
                .values({"Connected cells", "Transmitters and Constructors"})
                .tooltip(Const::GenomeTransmitterEnergyDistributionTooltip)
                .textWidth(CellTypeTextWidth),
            transmitter._mode);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processMuscleContent(MuscleDescription& muscle)
{
    if (ImGui::TreeNodeEx("Properties###muscle", TreeNodeFlags)) {
        //AlienImGui::Combo(
        //    AlienImGui::ComboParameters()
        //        .name("Mode")
        //        .values({"Movement to sensor target", "Expansion and contraction", "Bending"})
        //        .textWidth(CellTypeTextWidth)
        //        .tooltip(Const::GenomeMuscleModeTooltip),
        //    muscle._mode);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processSensorContent(SensorDescription& sensor)
{
    if (ImGui::TreeNodeEx("Properties###sensor", TreeNodeFlags)) {
        int constructorMode = sensor._autoTriggerInterval == 0 ? 0 : 1;
        if (AlienGui::Combo(
                AlienGui::ComboParameters()
                    .name("Activation mode")
                    .textWidth(CellTypeTextWidth)
                    .values({"Manual", "Automatic"})
                    .tooltip(Const::GenomeConstructorActivationModeTooltip),
                constructorMode)) {
            sensor._autoTriggerInterval = constructorMode;
        }
        if (constructorMode == 1) {
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Interval").textWidth(CellTypeTextWidth).tooltip(Const::GenomeConstructorIntervalTooltip),
                sensor._autoTriggerInterval);
        }

        AlienGui::ComboOptionalColor(
            AlienGui::ComboColorParameters().name("Scan color").textWidth(CellTypeTextWidth).tooltip(Const::GenomeSensorScanColorTooltip),
            sensor._restrictToColor);

        AlienGui::Combo(
            AlienGui::ComboParameters()
                .name("Scan mutants")
                .values({"None", "Same mutants", "Other mutants", "Free cells", "Handcrafted cells", "Less complex mutants", "More complex mutants"})
                .textWidth(CellTypeTextWidth)
                .tooltip(Const::SensorRestrictToMutantsTooltip),
            sensor._restrictToMutants);
        AlienGui::InputFloat(
            AlienGui::InputFloatParameters()
                .name("Min density")
                .format("%.2f")
                .step(0.05f)
                .textWidth(CellTypeTextWidth)
                .tooltip(Const::GenomeSensorMinDensityTooltip),
            sensor._minDensity);
        AlienGui::InputOptionalInt(
            AlienGui::InputIntParameters().name("Min range").textWidth(CellTypeTextWidth).tooltip(Const::GenomeSensorMinRangeTooltip), sensor._minRange);
        AlienGui::InputOptionalInt(
            AlienGui::InputIntParameters().name("Max range").textWidth(CellTypeTextWidth).tooltip(Const::GenomeSensorMaxRangeTooltip), sensor._maxRange);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processReconnectorContent(ReconnectorDescription& reconnector)
{
    if (ImGui::TreeNodeEx("Properties###reconnector", TreeNodeFlags)) {
        AlienGui::ComboOptionalColor(
            AlienGui::ComboColorParameters().name("Restrict to color").textWidth(CellTypeTextWidth).tooltip(Const::GenomeReconnectorRestrictToColorTooltip),
            reconnector._restrictToColor);
        AlienGui::Combo(
            AlienGui::ComboParameters()
                .name("Restrict to mutants")
                .values({"None", "Same mutants", "Other mutants", "Free cells", "Handcrafted cells", "Less complex mutants", "More complex mutants"})
                .textWidth(CellTypeTextWidth)
                .tooltip(Const::ReconnectorRestrictToMutantsTooltip),
            reconnector._restrictToMutants);

        ImGui::TreePop();
    }
}

void _InspectorWindow::processDetonatorContent(DetonatorDescription& detonator)
{
    if (ImGui::TreeNodeEx("Properties###detonator", TreeNodeFlags)) {
        AlienGui::Combo(
            AlienGui::ComboParameters()
                .name("State")
                .values({"Ready", "Activated", "Exploded"})
                .textWidth(CellTypeTextWidth)
                .tooltip(Const::DetonatorStateTooltip),
            detonator._state);

        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Countdown").textWidth(CellTypeTextWidth).tooltip(Const::GenomeDetonatorCountdownTooltip),
            detonator._countdown);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processParticle(ParticleDescription particle)
{
    auto origParticle = particle;
    auto energy = toFloat(particle._energy);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").textWidth(ParticleContentTextWidth), energy);

    particle._energy = energy;
    if (particle != origParticle) {
        _simulationFacade->changeParticle(particle);
    }
}

float _InspectorWindow::calcWindowWidth() const
{
    if (isCell()) {
        return StyleRepository::get().scale(CellWindowWidth);
    } else {
        return StyleRepository::get().scale(ParticleWindowWidth);
    }
}

void _InspectorWindow::validateAndCorrect(CellDescription& cell) const
{
    auto const& parameters = _simulationFacade->getSimulationParameters();

    cell._stiffness = std::max(0.0f, std::min(1.0f, cell._stiffness));
    cell._energy = std::max(0.0f, cell._energy);
    switch (cell.getCellType()) {
    case CellType_Constructor: {
        auto& constructor = std::get<ConstructorDescription>(cell._cellTypeData);
        auto numNodes = GenomeDescriptionConverterService::get().convertNodeAddressToNodeIndex(constructor._genome, toInt(constructor._genome.size()));
        if (numNodes > 0) {
            constructor._genomeCurrentNodeIndex = ((constructor._genomeCurrentNodeIndex % numNodes) + numNodes) % numNodes;
        } else {
            constructor._genomeCurrentNodeIndex = 0;
        }

        auto numRepetitions = GenomeDescriptionConverterService::get().getNumRepetitions(constructor._genome);
        if (numRepetitions != std::numeric_limits<int>::max()) {
            constructor._genomeCurrentRepetition = ((constructor._genomeCurrentRepetition % numRepetitions) + numRepetitions) % numRepetitions;
        } else {
            constructor._genomeCurrentRepetition = 0;
        }

        constructor._constructionActivationTime = ((constructor._constructionActivationTime % MAX_ACTIVATION_TIME) + MAX_ACTIVATION_TIME) % MAX_ACTIVATION_TIME;
        if (constructor._constructionActivationTime < 0) {
            constructor._constructionActivationTime = 0;
        }
        if (constructor._autoTriggerInterval < 0) {
            constructor._autoTriggerInterval = 0;
        }
        constructor._genomeGeneration = std::max(0, constructor._genomeGeneration);
    } break;
    case CellType_Sensor: {
        auto& sensor = std::get<SensorDescription>(cell._cellTypeData);
        sensor._minDensity = std::max(0.0f, std::min(1.0f, sensor._minDensity));
        if (sensor._minRange) {
            sensor._minRange = std::max(0, std::min(127, *sensor._minRange));
        }
        if (sensor._maxRange) {
            sensor._maxRange = std::max(0, std::min(127, *sensor._maxRange));
        }
    } break;
    case CellType_Oscillator: {
        auto& oscillator = std::get<OscillatorDescription>(cell._cellTypeData);
        oscillator._autoTriggerInterval = std::max(0, oscillator._autoTriggerInterval);
        oscillator._alternationInterval = std::max(0, oscillator._alternationInterval);
    } break;
    case CellType_Detonator: {
        auto& detonator = std::get<DetonatorDescription>(cell._cellTypeData);
        detonator._countdown = std::min(0xffff, std::max(0, detonator._countdown));
    } break;
    }
}
