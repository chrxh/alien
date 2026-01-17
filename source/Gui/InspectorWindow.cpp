#include "InspectorWindow.h"

#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include <imgui.h>

#include <EngineInterface/DescriptionEditService.h>
#include <EngineInterface/SimulationFacade.h>

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
    auto constexpr CellWindowWidth = 380.0f;
    auto constexpr ParticleWindowWidth = 280.0f;
    auto constexpr BaseTabTextWidth = 162.0f;
    auto constexpr CellTypeTextWidth = 195.0f;
    auto constexpr CellTypeDefenderWidth = 100.0f;
    auto constexpr CellTypeBaseTabTextWidth = 150.0f;
    auto constexpr SignalTextWidth = 130.0f;
    auto constexpr GenomeTabTextWidth = 195.0f;
    auto constexpr ParticleContentTextWidth = 80.0f;

    auto constexpr TreeNodeFlags = ImGuiTreeNodeFlags_FramePadding | ImGuiTreeNodeFlags_DefaultOpen;
}

_InspectorWindow::_InspectorWindow(uint64_t entityId, RealVector2D const& initialPos, bool selectGenomeTab)
    : _entityId(entityId)
    , _initialPos(initialPos)
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
    auto borderlessRendering = _SimulationFacade::get()->getSimulationParameters().borderlessRendering.value;
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    auto entity = EditorModel::get().getInspectedEntity(_entityId);
    if (ImGui::Begin(generateTitle().c_str(), &_on, ImGuiWindowFlags_HorizontalScrollbar)) {
        auto windowPos = ImGui::GetWindowPos();
        if (isCell()) {
            processCell(std::get<ExtendedObjectDescription>(entity));
        } else {
            processParticle(std::get<EnergyDescription>(entity));
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
    return std::holds_alternative<ExtendedObjectDescription>(entity);
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

void _InspectorWindow::processCell(ExtendedObjectDescription& extendedCell)
{
    if (ImGui::BeginTabBar("##CellInspect", /*ImGuiTabBarFlags_AutoSelectNewTabs | */ ImGuiTabBarFlags_FittingPolicyResizeDown)) {
        auto& object = extendedCell.object;
        auto origCell = object;
        processCellGeneralTab(extendedCell);
        processCellTypeTab(object);
        processCellTypePropertiesTab(object);
        if (object.getCellRef().getCellType() == CellType_Constructor) {
            processCellGenomeTab(std::get<ConstructorDescription>(object.getCellRef()._cellType));
        }
        if (object.getCellRef().getCellType() == CellType_Injector) {
            processCellGenomeTab(std::get<InjectorDescription>(object.getCellRef()._cellType));
        }

        validateAndCorrect(object);

        ImGui::EndTabBar();

        if (object != origCell) {
            _SimulationFacade::get()->changeCell(object);
        }
    }
}

void _InspectorWindow::processCellGeneralTab(ExtendedObjectDescription& extendedCell)
{
    if (ImGui::BeginTabItem("General", nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            auto& object = extendedCell.object;
            auto& genome = extendedCell.genome;
            if (ImGui::TreeNodeEx("Properties###general", TreeNodeFlags)) {
                if (extendedCell.genome.has_value()) {
                    if (AlienGui::Button("Edit genome")) {
                        GenomeEditorWindow::get().openTab(extendedCell.creatureId.value(), genome.value());
                    }
                }

                std::stringstream ss;
                ss << "0x" << std::hex << std::uppercase << object._id;
                auto objectId = ss.str();

                AlienGui::ComboColor(
                    AlienGui::ComboColorParameters().name("Color").textWidth(BaseTabTextWidth).tooltip(Const::GenomeColorTooltip), object._color);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Usable energy").format("%.2f").textWidth(BaseTabTextWidth).tooltip(Const::CellEnergyTooltip), object.getCellRef()._usableEnergy);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Raw energy").format("%.2f").textWidth(BaseTabTextWidth).tooltip(Const::CellEnergyTooltip),
                    object.getCellRef()._rawEnergy);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Age").textWidth(BaseTabTextWidth).tooltip(Const::CellAgeTooltip), object.getCellRef()._age);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Position X").format("%.2f").textWidth(BaseTabTextWidth), object._pos.x);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Position Y").format("%.2f").textWidth(BaseTabTextWidth), object._pos.y);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Velocity X").format("%.2f").textWidth(BaseTabTextWidth), object._vel.x);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Velocity Y").format("%.2f").textWidth(BaseTabTextWidth), object._vel.y);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters()
                        .name("Stiffness")
                        .format("%.2f")
                        .step(0.05f)
                        .textWidth(BaseTabTextWidth)
                        .tooltip(Const::CellStiffnessTooltip),
                    object._stiffness);
                AlienGui::Checkbox(
                    AlienGui::CheckboxParameters().name("Sticky").textWidth(BaseTabTextWidth).tooltip(Const::CellIndestructibleTooltip), object._sticky);
                AlienGui::Checkbox(
                    AlienGui::CheckboxParameters().name("Indestructible wall").textWidth(BaseTabTextWidth).tooltip(Const::CellIndestructibleTooltip),
                    object._fixed);
                AlienGui::InputText(
                    AlienGui::InputTextParameters().name("Cell id").textWidth(BaseTabTextWidth).tooltip(Const::CellIdTooltip).readOnly(true), objectId);
                if (auto frontAngle = object.getCellRef()._frontAngle) {
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("TEMP: front angle").format("%.1f").textWidth(BaseTabTextWidth), frontAngle.value());
                    object.getCellRef()._frontAngle = frontAngle;
                }
                if (object.getCellRef().getCellType() == CellType_Muscle) {
                    auto& muscle = std::get<MuscleDescription>(object.getCellRef()._cellType);
                    if (muscle.getMode() == MuscleMode_AutoBending) {
                        auto& bending = std::get<AutoBendingDescription>(muscle._mode);
                        if (auto initialAngle = bending._initialAngle) {
                            AlienGui::InputFloat(
                                AlienGui::InputFloatParameters().name("TEMP: initial angle").format("%.1f").textWidth(BaseTabTextWidth), initialAngle.value());
                            bending._initialAngle = initialAngle;
                        }
                    }
                    if (muscle.getMode() == MuscleMode_ManualBending) {
                        auto& bending = std::get<ManualBendingDescription>(muscle._mode);
                        if (auto initialAngle = bending._initialAngle) {
                            AlienGui::InputFloat(
                                AlienGui::InputFloatParameters().name("TEMP: initial angle").format("%.1f").textWidth(BaseTabTextWidth), initialAngle.value());
                            bending._initialAngle = initialAngle;
                        }
                    }
                }
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Signal routing", TreeNodeFlags)) {
                int modeAsInt = static_cast<int>(object.getCellRef()._signalRestriction._mode);
                if (AlienGui::Switcher(
                    AlienGui::SwitcherParameters().name("Signal restriction").values(Const::SignalRestrictionModeStrings).textWidth(BaseTabTextWidth),
                    modeAsInt)) {
                    object.getCellRef()._signalRestriction._mode = static_cast<SignalRestrictionMode>(modeAsInt);
                }
                bool restrictionActive = (object.getCellRef()._signalRestriction._mode == SignalRestrictionMode_Active || 
                                          object.getCellRef()._signalRestriction._mode == SignalRestrictionMode_Conditional);
                if (restrictionActive) {
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Signal base angle").format("%.1f").step(2.0f).textWidth(BaseTabTextWidth),
                        object.getCellRef()._signalRestriction._baseAngle);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Signal opening angle").format("%.1f").step(2.0f).textWidth(BaseTabTextWidth),
                        object.getCellRef()._signalRestriction._openingAngle);
                }
                ImGui::TreePop();
            }

            if (ImGui::TreeNodeEx("Associated creature##Base", TreeNodeFlags)) {
                if (genome.has_value()) {
                    std::stringstream ss;
                    ss << "0x" << std::hex << std::uppercase << extendedCell.creatureId.value();
                    auto creatureId = ss.str();
                    AlienGui::InputText(
                        AlienGui::InputTextParameters().name("Creature id").textWidth(BaseTabTextWidth).tooltip(Const::CellIdTooltip).readOnly(true),
                        creatureId);
                }
                ImGui::TreePop();
            }
            if (ImGui::TreeNodeEx("Genome", TreeNodeFlags)) {
                if (genome.has_value()) {
                    AlienGui::InputInt(
                        AlienGui::InputIntParameters().name("Node index").textWidth(BaseTabTextWidth), object.getCellRef()._nodeIndex);
                }
                ImGui::TreePop();
            }
            if (ImGui::TreeNodeEx("Connections to other cells", TreeNodeFlags)) {
                for (auto const& [index, connection] : object._connections | boost::adaptors::indexed(0)) {
                    if (ImGui::TreeNodeEx(("Connection [" + std::to_string(index) + "]").c_str(), ImGuiTreeNodeFlags_None)) {
                        std::stringstream ss;
                        ss << "0x" << std::hex << std::uppercase << connection._objectId;
                        auto objectId = ss.str();

                        AlienGui::InputText(
                            AlienGui::InputTextParameters().name("Cell id").textWidth(BaseTabTextWidth).tooltip(Const::CellIdTooltip).readOnly(true), objectId);
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

void _InspectorWindow::processCellTypeTab(ObjectDescription& object)
{
    if (ImGui::BeginTabItem("Function", nullptr, ImGuiTabItemFlags_None)) {
        int type = object.getCellRef().getCellType();
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

            processNeuronContent(object);

            if (ImGui::TreeNodeEx("Properties###type", TreeNodeFlags)) {
                if (AlienGui::Combo(
                        AlienGui::ComboParameters()
                            .name("Function")
                            .values(Const::CellTypeStrings)
                            .textWidth(CellTypeBaseTabTextWidth)
                            .tooltip(Const::getCellTypeTooltip(type)),
                        type)) {
                    switch (type) {
                    case CellType_Base: {
                        object.getCellRef()._cellType = BaseDescription();
                    } break;
                    case CellType_Depot: {
                        object.getCellRef()._cellType = DepotDescription();
                    } break;
                    case CellType_Constructor: {
                        object.getCellRef()._cellType = ConstructorDescription();
                    } break;
                    case CellType_Sensor: {
                        object.getCellRef()._cellType = SensorDescription();
                    } break;
                    case CellType_Generator: {
                        object.getCellRef()._cellType = GeneratorDescription();
                    } break;
                    case CellType_Attacker: {
                        object.getCellRef()._cellType = AttackerDescription();
                    } break;
                    case CellType_Injector: {
                        object.getCellRef()._cellType = InjectorDescription();
                    } break;
                    case CellType_Muscle: {
                        object.getCellRef()._cellType = MuscleDescription();
                    } break;
                    case CellType_Defender: {
                        object.getCellRef()._cellType = DefenderDescription();
                    } break;
                    case CellType_Reconnector: {
                        object.getCellRef()._cellType = ReconnectorDescription();
                    } break;
                    case CellType_Detonator: {
                        object.getCellRef()._cellType = DetonatorDescription();
                    } break;
                    }
                }

                AlienGui::InputInt(
                    AlienGui::InputIntParameters()
                        .name("Activation time")
                        .textWidth(CellTypeBaseTabTextWidth)
                        .tooltip(Const::GenomeConstructorOffspringActivationTime),
                    object.getCellRef()._activationTime);
                AlienGui::Combo(
                    AlienGui::ComboParameters()
                        .name("Living state")
                        .textWidth(CellTypeBaseTabTextWidth)
                        .values({"Ready", "Under construction", "Activating", "Detached", "Reviving", "Dying"})
                        .tooltip(Const::CellCellStateTooltip),
                    object.getCellRef()._cellState);
                ImGui::TreePop();
            }
        }
        if (object.getCellRef()._signalState == SignalState_Active) {
            if (ImGui::TreeNodeEx("Signals", TreeNodeFlags)) {
                int index = 0;
                for (auto& channel : object.getCellRef()._signal._channels) {
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

void _InspectorWindow::processCellTypePropertiesTab(ObjectDescription& object)
{
    if (object.getObjectType() == ObjectType_Structure || object.getObjectType() == ObjectType_FreeCell) {
        return;
    }

    std::string title = Const::CellTypeStrings.at(object.getCellRef().getCellType());
    if (ImGui::BeginTabItem(title.c_str(), nullptr, ImGuiTabItemFlags_None)) {
        if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {
            switch (object.getCellRef().getCellType()) {
            case CellType_Base: {
            } break;
            case CellType_Depot: {
                processDepotContent(std::get<DepotDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Constructor: {
                processConstructorContent(std::get<ConstructorDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Sensor: {
                processSensorContent(std::get<SensorDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Generator: {
                processGeneratorContent(std::get<GeneratorDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Attacker: {
                processAttackerContent(std::get<AttackerDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Injector: {
                processInjectorContent(std::get<InjectorDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Muscle: {
                processMuscleContent(std::get<MuscleDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Defender: {
                processDefenderContent(std::get<DefenderDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Reconnector: {
                processReconnectorContent(std::get<ReconnectorDescription>(object.getCellRef()._cellType));
            } break;
            case CellType_Detonator: {
                processDetonatorContent(std::get<DetonatorDescription>(object.getCellRef()._cellType));
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
    auto const& parameters = _SimulationFacade::get()->getSimulationParameters();

    int flags = ImGuiTabItemFlags_None;
    if (_selectGenomeTab) {
        flags = flags | ImGuiTabItemFlags_SetSelected;
        _selectGenomeTab = false;
    }
    if (ImGui::BeginTabItem("Genome", nullptr, flags)) {
        //if (ImGui::BeginChild("##", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar)) {

        //    auto previewNodeResult = ImGui::TreeNodeEx("Preview (reference configuration)", TreeNodeFlags);
        //    AlienGui::HelpMarker(Const::GenomePreviewTooltip);
        //    if (previewNodeResult) {
        //        if (ImGui::BeginChild("##child", ImVec2(0, scale(200)), true, ImGuiWindowFlags_HorizontalScrollbar)) {
        //            auto genomDesc = GenomeDescriptionConverterService::get().convertBytesToDescription(desc._genome);
        //            auto previewDesc = PreviewDescriptionConverterService::get().convert(genomDesc, std::nullopt, parameters);
        //            std::optional<int> selectedNodeDummy;
        //            AlienGui::ShowPreviewDescription(previewDesc, _genomeZoom, selectedNodeDummy);
        //        }
        //        ImGui::EndChild();
        //        if (AlienGui::Button("Edit")) {
        //            GenomeEditorWindow::get().openTab(GenomeDescriptionConverterService::get().convertBytesToDescription(desc._genome));
        //        }

        //        ImGui::SameLine();
        //        if (AlienGui::Button(AlienGui::ButtonParameters().buttonText("Inject from editor").textWidth(ImGui::GetContentRegionAvail().x))) {
        //            printOverlayMessage("Genome injected");
        //            desc._genome = GenomeDescriptionConverterService::get().convertDescriptionToBytes(GenomeEditorWindow::get().getCurrentGenome());
        //            if constexpr (std::is_same<Description, ConstructorDescription>()) {
        //                desc._currentNodeIndex = 0;
        //                desc.numExpectedCells(0);
        //            }
        //        }
        //        ImGui::TreePop();
        //    }

        //    if (ImGui::TreeNodeEx("Properties (entire genome)", TreeNodeFlags)) {
        //        auto numNodes = toInt(GenomeDescriptionConverterService::get().getNumNodesRecursively(desc._genome, true));
        //        AlienGui::InputInt(
        //            AlienGui::InputIntParameters()
        //                .name("Number of cells")
        //                .textWidth(GenomeTabTextWidth)
        //                .readOnly(true)
        //                .tooltip(Const::GenomeNumCellsRecursivelyTooltip),
        //            numNodes);

        //        auto numBytes = toInt(desc._genome.size());
        //        AlienGui::InputInt(
        //            AlienGui::InputIntParameters().name("Bytes").textWidth(GenomeTabTextWidth).readOnly(true).tooltip(Const::GenomeBytesTooltip), numBytes);

        //        AlienGui::InputInt(
        //            AlienGui::InputIntParameters().name("Generation").textWidth(GenomeTabTextWidth).tooltip(Const::GenomeGenerationTooltip),
        //            desc._generation);
        //        ImGui::TreePop();
        //    }

        //    if (ImGui::TreeNodeEx("Properties (principal genome part)", TreeNodeFlags)) {

        //        auto genomeDesc = GenomeDescriptionConverterService::get().convertBytesToDescription(desc._genome);
        //        auto numBranches = genomeDesc._header.getNumBranches();
        //        AlienGui::InputInt(
        //            AlienGui::InputIntParameters()
        //                .name("Number of branches")
        //                .textWidth(GenomeTabTextWidth)
        //                .readOnly(true)
        //                .tooltip(Const::GenomeNumBranchesTooltip),
        //            numBranches);

        //        auto numRepetitions = genomeDesc._header._numRepetitions;
        //        AlienGui::InputInt(
        //            AlienGui::InputIntParameters()
        //                .name("Repetitions per branch")
        //                .textWidth(GenomeTabTextWidth)
        //                .infinity(true)
        //                .readOnly(true)
        //                .tooltip(Const::GenomeRepetitionsPerBranchTooltip),
        //            numRepetitions);

        //        auto numNodes = toInt(genomeDesc._objects.size());
        //        AlienGui::InputInt(
        //            AlienGui::InputIntParameters()
        //                .name("Cells per repetition")
        //                .textWidth(GenomeTabTextWidth)
        //                .readOnly(true)
        //                .tooltip(Const::GenomeNumCellsTooltip),
        //            numNodes);

        //        if constexpr (std::is_same<Description, ConstructorDescription>()) {
        //            AlienGui::InputInt(
        //                AlienGui::InputIntParameters().name("Current branch index").textWidth(GenomeTabTextWidth).tooltip(Const::GenomeCurrentBranchTooltip),
        //                desc._currentBranch);
        //            AlienGui::InputInt(
        //                AlienGui::InputIntParameters()
        //                    .name("Current repetition index")
        //                    .textWidth(GenomeTabTextWidth)
        //                    .tooltip(Const::GenomeCurrentConcatenationTooltip),
        //                desc._currentConcatenation);
        //            AlienGui::InputInt(
        //                AlienGui::InputIntParameters().name("Current cell index").textWidth(GenomeTabTextWidth).tooltip(Const::GenomeCurrentObjectTooltip),
        //                desc._currentNodeIndex);
        //        }
        //        ImGui::TreePop();
        //    }
        //}
        ImGui::EndChild();
        ImGui::EndTabItem();
    }
}


void _InspectorWindow::processGeneratorContent(GeneratorDescription& _generator)
{
    if (ImGui::TreeNodeEx("Properties###_generator", TreeNodeFlags)) {

        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Pulse interval").textWidth(CellTypeTextWidth).tooltip(Const::GenomeGeneratorPulseIntervalTooltip),
            _generator._autoTriggerInterval);
        bool alternation = _generator._alternationInterval > 0;
        if (AlienGui::Checkbox(
                AlienGui::CheckboxParameters().name("Alternating pulses").textWidth(CellTypeTextWidth).tooltip(Const::GenomeGeneratorAlternatingPulsesTooltip),
                alternation)) {
            _generator._alternationInterval = alternation ? 1 : 0;
        }
        if (alternation) {
            AlienGui::InputInt(
                AlienGui::InputIntParameters().name("Pulses per phase").textWidth(CellTypeTextWidth).tooltip(Const::GenomeGeneratorPulsesPerPhaseTooltip),
                _generator._alternationInterval);
        }
        ImGui::TreePop();
    }
}

void _InspectorWindow::processNeuronContent(ObjectDescription& object)
{
    if (ImGui::TreeNodeEx("Neural network", TreeNodeFlags)) {
        //AlienGui::NeuralNetEditor(
        //    AlienGui::NeuralNetEditorParameters().rightMargin(0),
        //    object.getCellRef()._neuralNetwork->_weights,
        //    object.getCellRef()._neuralNetwork->_biases,
        //    object.getCellRef()._neuralNetwork->_activationFunctions);
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
            AlienGui::InputFloatParameters().name("Construction angle").textWidth(CellTypeTextWidth).format("%.1f").tooltip("Angle for construction direction"),
            constructor._constructionAngle);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processInjectorContent(InjectorDescription& injector)
{
    if (ImGui::TreeNodeEx("Properties###injector", TreeNodeFlags)) {
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Gene index").textWidth(CellTypeTextWidth), injector._geneIndex);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processAttackerContent(AttackerDescription& attacker)
{
    if (ImGui::TreeNodeEx("Properties###attacker", TreeNodeFlags)) {
        auto mode = attacker.getMode();
        ImGui::Text("Mode: %s", Const::AttackerModeStrings.at(mode).c_str());

        if (mode == AttackerMode_FreeCell) {
            auto& attackFreeCell = std::get<AttackFreeCellDescription>(attacker._mode);
            if (attackFreeCell._restrictToColor.has_value()) {
                ImGui::Text("Restrict to color: %d", *attackFreeCell._restrictToColor);
            }
        } else if (mode == AttackerMode_Creature) {
            auto& attackCreature = std::get<AttackCreatureDescription>(attacker._mode);
            if (attackCreature._minNumCells.has_value()) {
                ImGui::Text("Min creature cells: %d", *attackCreature._minNumCells);
            }
            if (attackCreature._maxNumCells.has_value()) {
                ImGui::Text("Max creature cells: %d", *attackCreature._maxNumCells);
            }
            if (attackCreature._restrictToColor.has_value()) {
                ImGui::Text("Restrict to color: %d", *attackCreature._restrictToColor);
            }
        }
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

void _InspectorWindow::processDepotContent(DepotDescription& transmitter)
{
    if (ImGui::TreeNodeEx("Properties###depot", TreeNodeFlags)) {
        ImGui::TreePop();
    }
}

void _InspectorWindow::processMuscleContent(MuscleDescription& muscle)
{
    if (ImGui::TreeNodeEx("Properties###muscle", TreeNodeFlags)) {
        //AlienGui::Combo(
        //    AlienGui::ComboParameters()
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

        // Mode selection
        auto mode = sensor.getMode();
        AlienGui::Combo(
            AlienGui::ComboParameters().name("Mode").values(Const::SensorModeStrings).textWidth(CellTypeTextWidth),
            mode);
        // Note: Mode cannot be changed in inspector - only viewing
        
        // Mode-specific parameters
        if (mode == SensorMode_DetectEnergy) {
            auto& detectEnergy = std::get<DetectEnergyDescription>(sensor._mode);
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters()
                    .name("Min density")
                    .format("%.2f")
                    .step(0.05f)
                    .textWidth(CellTypeTextWidth)
                    .tooltip(Const::GenomeSensorMinDensityTooltip),
                detectEnergy._minDensity);
        } else if (mode == SensorMode_DetectStructure) {
            // No parameters
        } else if (mode == SensorMode_DetectFreeCell) {
            auto& detectFreeCell = std::get<DetectFreeCellDescription>(sensor._mode);
            AlienGui::InputFloat(
                AlienGui::InputFloatParameters()
                    .name("Min density")
                    .format("%.2f")
                    .step(0.05f)
                    .textWidth(CellTypeTextWidth)
                    .tooltip(Const::GenomeSensorMinDensityTooltip),
                detectFreeCell._minDensity);
            AlienGui::ComboOptionalColor(
                AlienGui::ComboColorParameters().name("Restrict to color").textWidth(CellTypeTextWidth).tooltip(Const::GenomeSensorScanColorTooltip),
                detectFreeCell._restrictToColor);
        } else if (mode == SensorMode_DetectCreature) {
            auto& detectCreature = std::get<DetectCreatureDescription>(sensor._mode);
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Min num cells").textWidth(CellTypeTextWidth),
                detectCreature._minNumCells);
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Max num cells").textWidth(CellTypeTextWidth),
                detectCreature._maxNumCells);
            AlienGui::ComboOptionalColor(
                AlienGui::ComboColorParameters().name("Restrict to color").textWidth(CellTypeTextWidth).tooltip(Const::GenomeSensorScanColorTooltip),
                detectCreature._restrictToColor);
            AlienGui::Combo(
                AlienGui::ComboParameters()
                    .name("Restrict to lineage")
                    .values({"No", "Same lineage", "Other lineage"})
                    .textWidth(CellTypeTextWidth),
                detectCreature._restrictToLineage);
        }

        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Min range").textWidth(CellTypeTextWidth).tooltip(Const::GenomeSensorMinRangeTooltip), sensor._minRange);
        AlienGui::InputInt(
            AlienGui::InputIntParameters().name("Max range").textWidth(CellTypeTextWidth).tooltip(Const::GenomeSensorMaxRangeTooltip), sensor._maxRange);
        ImGui::TreePop();
    }
}

void _InspectorWindow::processReconnectorContent(ReconnectorDescription& reconnector)
{
    if (ImGui::TreeNodeEx("Properties###reconnector", TreeNodeFlags)) {
        // Mode selection
        auto mode = reconnector.getMode();
        AlienGui::Combo(
            AlienGui::ComboParameters().name("Mode").values(Const::ReconnectorModeStrings).textWidth(CellTypeTextWidth),
            mode);
        // Note: Mode cannot be changed in inspector - only viewing

        // Mode-specific parameters
        if (mode == ReconnectorMode_FreeCell) {
            auto& freeCell = std::get<ReconnectFreeCellDescription>(reconnector._mode);
            AlienGui::ComboOptionalColor(
                AlienGui::ComboColorParameters().name("Restrict to color").textWidth(CellTypeTextWidth).tooltip(Const::GenomeReconnectorRestrictToColorTooltip),
                freeCell._restrictToColor);
        } else if (mode == ReconnectorMode_Creature) {
            auto& creature = std::get<ReconnectCreatureDescription>(reconnector._mode);
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Min creature cells").textWidth(CellTypeTextWidth),
                creature._minNumCells);
            AlienGui::InputOptionalInt(
                AlienGui::InputIntParameters().name("Max creature cells").textWidth(CellTypeTextWidth),
                creature._maxNumCells);
            AlienGui::ComboOptionalColor(
                AlienGui::ComboColorParameters().name("Restrict to color").textWidth(CellTypeTextWidth).tooltip(Const::GenomeReconnectorRestrictToColorTooltip),
                creature._restrictToColor);
            AlienGui::Combo(
                AlienGui::ComboParameters()
                    .name("Restrict to lineage")
                    .values({"No", "Same lineage", "Other lineage"})
                    .textWidth(CellTypeTextWidth),
                creature._restrictToLineage);
        }

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

void _InspectorWindow::processParticle(EnergyDescription particle)
{
    auto origParticle = particle;
    auto energy = toFloat(particle._energy);
    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").textWidth(ParticleContentTextWidth), energy);

    particle._energy = energy;
    if (particle != origParticle) {
        _SimulationFacade::get()->changeParticle(particle);
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

void _InspectorWindow::validateAndCorrect(ObjectDescription& object) const
{
    object._stiffness = std::max(0.0f, std::min(1.0f, object._stiffness));
    object.getCellRef()._usableEnergy = std::max(0.0f, object.getCellRef()._usableEnergy);
    switch (object.getCellRef().getCellType()) {
    case CellType_Constructor: {
        auto& constructor = std::get<ConstructorDescription>(object.getCellRef()._cellType);
        //auto numNodes = GenomeDescriptionConverterService::get().convertNodeAddressToNodeIndex(constructor._genome, toInt(constructor._genome.size()));
        //if (numNodes > 0) {
        //    constructor._currentNodeIndex = ((constructor._currentNodeIndex % numNodes) + numNodes) % numNodes;
        //} else {
        //    constructor._currentNodeIndex = 0;
        //}

        //auto numRepetitions = GenomeDescriptionConverterService::get().getNumRepetitions(constructor._genome);
        //if (numRepetitions != std::numeric_limits<int>::max()) {
        //    constructor._currentConcatenation = ((constructor._currentConcatenation % numRepetitions) + numRepetitions) % numRepetitions;
        //} else {
        //    constructor._currentConcatenation = 0;
        //}

        constructor._constructionActivationTime = ((constructor._constructionActivationTime % MAX_ACTIVATION_TIME) + MAX_ACTIVATION_TIME) % MAX_ACTIVATION_TIME;
        if (constructor._constructionActivationTime < 0) {
            constructor._constructionActivationTime = 0;
        }
        if (constructor._autoTriggerInterval < 0) {
            constructor._autoTriggerInterval = 0;
        }
        //constructor._generation = std::max(0, constructor._generation);
    } break;
    case CellType_Sensor: {
        auto& sensor = std::get<SensorDescription>(object.getCellRef()._cellType);
        auto mode = sensor.getMode();
        if (mode == SensorMode_DetectEnergy) {
            auto& detectEnergy = std::get<DetectEnergyDescription>(sensor._mode);
            detectEnergy._minDensity = std::max(0.0f, std::min(1.0f, detectEnergy._minDensity));
        } else if (mode == SensorMode_DetectFreeCell) {
            auto& detectFreeCell = std::get<DetectFreeCellDescription>(sensor._mode);
            detectFreeCell._minDensity = std::max(0.0f, std::min(1.0f, detectFreeCell._minDensity));
        }
        sensor._minRange = std::max(0, std::min(255, sensor._minRange));
        sensor._maxRange = std::max(0, std::min(255, sensor._maxRange));
    } break;
    case CellType_Generator: {
        auto& _generator = std::get<GeneratorDescription>(object.getCellRef()._cellType);
        _generator._autoTriggerInterval = std::max(0, _generator._autoTriggerInterval);
        _generator._alternationInterval = std::max(0, _generator._alternationInterval);
    } break;
    case CellType_Detonator: {
        auto& detonator = std::get<DetonatorDescription>(object.getCellRef()._cellType);
        detonator._countdown = std::min(0xffff, std::max(0, detonator._countdown));
    } break;
    }
}
