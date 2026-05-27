#include "InspectionWindow.h"

#include <iomanip>
#include <sstream>
#include <variant>

#include <imgui.h>

#include <EngineInterface/CellTypeConstants.h>
#include <EngineInterface/DescEditService.h>
#include <EngineInterface/EngineConstants.h>
#include <EngineInterface/SimulationFacade.h>

#include "AlienGui.h"
#include "EditorModel.h"
#include "GenomeEditorWindow.h"
#include "NeuralNetEditorWidget.h"
#include "SignalsBufferDialog.h"
#include "StyleRepository.h"
#include "Viewport.h"

using namespace std::string_literals;

float _InspectionWindow::_savedScrollY = 0;

namespace
{
    auto constexpr CellWindowWidth = 420.0f;
    auto constexpr ParticleWindowWidth = 320.0f;
    auto constexpr TableColumnWidth = 380.0f;
    auto constexpr TextWidth = 160.0f;

    std::string toHex(uint64_t id)
    {
        std::stringstream ss;
        ss << "0x" << std::hex << std::uppercase << id;
        return ss.str();
    }

    void inspectorHexId(std::string const& name, uint64_t id, float textWidth = TextWidth)
    {
        auto text = toHex(id);
        AlienGui::InputText(AlienGui::InputTextParameters().name(name).textWidth(textWidth).readOnly(true), text);
    }

    void inspectorText(std::string const& name, std::string const& value, float textWidth = TextWidth)
    {
        auto text = value;
        AlienGui::InputText(AlienGui::InputTextParameters().name(name).textWidth(textWidth).readOnly(true), text);
    }

    template <typename Func>
    void processPropertiesSubNode(std::string const& idSuffix, Func&& processWidgets)
    {
        if (AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Properties##" + idSuffix).rank(AlienGui::TreeNodeRank::Default).defaultOpen(true))) {
            processWidgets();
        }
        AlienGui::EndTreeNode();
    }

    ObjectTypeDesc createObjectTypeDesc(ObjectType type)
    {
        switch (type) {
        case ObjectType_Solid:
            return SolidDesc();
        case ObjectType_Fluid:
            return FluidDesc();
        case ObjectType_FreeCell:
            return FreeCellDesc();
        case ObjectType_Cell:
            return CellDesc();
        default:
            return CellDesc();
        }
    }

    CellTypeDesc createCellTypeDesc(CellType type)
    {
        switch (type) {
        case CellType_Base:
            return BaseDesc();
        case CellType_Depot:
            return DepotDesc();
        case CellType_Sensor:
            return SensorDesc();
        case CellType_Generator:
            return GeneratorDesc();
        case CellType_Attacker:
            return AttackerDesc();
        case CellType_Injector:
            return InjectorDesc();
        case CellType_Muscle:
            return MuscleDesc();
        case CellType_Defender:
            return DefenderDesc();
        case CellType_Reconnector:
            return ReconnectorDesc();
        case CellType_Detonator:
            return DetonatorDesc();
        case CellType_Digestor:
            return DigestorDesc();
        case CellType_Memory:
            return MemoryDesc();
        case CellType_Communicator:
            return CommunicatorDesc();
        case CellType_Void:
            return VoidDesc();
        default:
            return BaseDesc();
        }
    }

    SensorModeDesc createSensorModeDesc(SensorMode mode)
    {
        switch (mode) {
        case SensorMode_Telemetry:
            return TelemetryDesc();
        case SensorMode_DetectEnergy:
            return DetectEnergyDesc();
        case SensorMode_DetectSolid:
            return DetectSolidDesc();
        case SensorMode_DetectFreeCell:
            return DetectFreeCellDesc();
        case SensorMode_DetectCreature:
            return DetectCreatureDesc();
        default:
            return DetectCreatureDesc();
        }
    }

    GeneratorModeDesc createGeneratorModeDesc(GeneratorMode mode)
    {
        switch (mode) {
        case GeneratorMode_SquareSignal:
            return SquareSignalDesc();
        case GeneratorMode_SawtoothSignal:
            return SawtoothSignalDesc();
        default:
            return SquareSignalDesc();
        }
    }

    AttackerModeDesc createAttackerModeDesc(AttackerMode mode)
    {
        switch (mode) {
        case AttackerMode_FreeCell:
            return AttackFreeCellDesc();
        case AttackerMode_Creature:
            return AttackCreatureDesc();
        default:
            return AttackCreatureDesc();
        }
    }

    MuscleModeDesc createMuscleModeDesc(MuscleMode mode)
    {
        switch (mode) {
        case MuscleMode_AutoBending:
            return AutoBendingDesc();
        case MuscleMode_ManualBending:
            return ManualBendingDesc();
        case MuscleMode_AngleBending:
            return AngleBendingDesc();
        case MuscleMode_AutoCrawling:
            return AutoCrawlingDesc();
        case MuscleMode_ManualCrawling:
            return ManualCrawlingDesc();
        case MuscleMode_DirectMovement:
            return DirectMovementDesc();
        default:
            return AutoBendingDesc();
        }
    }

    ReconnectorModeDesc createReconnectorModeDesc(ReconnectorMode mode)
    {
        switch (mode) {
        case ReconnectorMode_Solid:
            return ReconnectSolidDesc();
        case ReconnectorMode_FreeCell:
            return ReconnectFreeCellDesc();
        case ReconnectorMode_Creature:
            return ReconnectCreatureDesc();
        default:
            return ReconnectCreatureDesc();
        }
    }

    MemoryModeDesc createMemoryModeDesc(MemoryMode mode)
    {
        switch (mode) {
        case MemoryMode_SignalDelay:
            return SignalDelayDesc();
        case MemoryMode_SignalRecorder:
            return SignalRecorderDesc();
        case MemoryMode_SignalStorage:
            return SignalStorageDesc();
        case MemoryMode_SignalIntegrator:
            return SignalIntegratorDesc();
        default:
            return SignalDelayDesc();
        }
    }

    CommunicatorModeDesc createCommunicatorModeDesc(CommunicatorMode mode)
    {
        switch (mode) {
        case CommunicatorMode_Sender:
            return SenderDesc();
        case CommunicatorMode_Receiver:
            return ReceiverDesc();
        default:
            return SenderDesc();
        }
    }
}

_InspectionWindow::_InspectionWindow(uint64_t entityId, RealVector2D const& initialPos, bool selectGenomeTab)
    : _entityId(entityId)
    , _initialPos(initialPos)
    , _selectGenomeTab(selectGenomeTab)
{
    _neuralNetWidget = _NeuralNetEditorWidget::create();
}

_InspectionWindow::~_InspectionWindow() {}

void _InspectionWindow::process()
{
    if (!_on) {
        return;
    }
    auto width = calcWindowWidth();
    auto height = isObject() ? StyleRepository::get().scale(500.0f) : StyleRepository::get().scale(180.0f);
    auto borderlessRendering = _SimulationFacade::get()->getSimulationParameters().borderlessRendering.value;
    ImGui::SetNextWindowBgAlpha(Const::WindowAlpha * ImGui::GetStyle().Alpha);
    ImGui::SetNextWindowSize({width, height}, ImGuiCond_Appearing);
    ImGui::SetNextWindowPos({_initialPos.x, _initialPos.y}, ImGuiCond_Appearing);
    auto entity = EditorModel::get().getInspectedEntity(_entityId);
    if (ImGui::Begin(generateTitle().c_str(), &_on, ImGuiWindowFlags_HorizontalScrollbar)) {
        if (_isFirstFrame && _savedScrollY > 0) {
            ImGui::SetScrollY(_savedScrollY);
        }
        auto windowPos = ImGui::GetWindowPos();
        if (isObject()) {
            auto extendedObject = std::get<ExtendedObjectDesc>(entity);
            processObject(extendedObject);
        } else {
            processParticle(std::get<EnergyDesc>(entity));
        }
        _savedScrollY = ImGui::GetScrollY();
        _isFirstFrame = false;
        ImDrawList* drawList = ImGui::GetBackgroundDrawList();
        auto entityPos = Viewport::get().mapWorldToViewPosition(DescEditService::get().getPos(entity), borderlessRendering);
        auto factor = StyleRepository::get().scale(1);

        drawList->AddLine({windowPos.x + 15.0f * factor, windowPos.y - 5.0f * factor}, {entityPos.x, entityPos.y}, Const::InspectorLineColor, 1.5f);
        drawList->AddRectFilled(
            {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor}, {windowPos.x + 25.0f * factor, windowPos.y}, Const::InspectorRectColor, 1.0, 0);
        drawList->AddRect(
            {windowPos.x + 5.0f * factor, windowPos.y - 10.0f * factor}, {windowPos.x + 25.0f * factor, windowPos.y}, Const::InspectorLineColor, 1.0, 0, 2.0f);
    }
    ImGui::End();
}

bool _InspectionWindow::isClosed() const
{
    return !_on;
}

uint64_t _InspectionWindow::getId() const
{
    return _entityId;
}

bool _InspectionWindow::isObject() const
{
    auto entity = EditorModel::get().getInspectedEntity(_entityId);
    return std::holds_alternative<ExtendedObjectDesc>(entity);
}

std::string _InspectionWindow::generateTitle() const
{
    std::stringstream ss;
    if (isObject()) {
        ss << "Cell with id 0x" << std::hex << std::uppercase << _entityId;
    } else {
        ss << "Energy particle with id 0x" << std::hex << std::uppercase << _entityId;
    }
    return ss.str();
}

void _InspectionWindow::processObject(ExtendedObjectDesc& extendedObject)
{
    auto& object = extendedObject.object;
    auto origObject = object;

    AlienGui::DynamicTableLayout table(TableColumnWidth);
    if (table.begin()) {
        processObjectNode(object);
        table.next();

        switch (object.getObjectType()) {
        case ObjectType_Solid:
            processSolidNode(object);
            table.next();
            break;
        case ObjectType_Fluid:
            processFluidNode(object);
            table.next();
            break;
        case ObjectType_FreeCell:
            processFreeCellNode(object);
            table.next();
            break;
        case ObjectType_Cell: {
            processCellNode(object);
            table.next();
            if (extendedObject.creature.has_value()) {
                processCreatureNode(extendedObject);
                table.next();
            }
            break;
        }
        default:
            break;
        }
        table.end();
    }

    validateAndCorrect(object);

    applyPendingSignalEntries(extendedObject);
    if (object != origObject) {
        _SimulationFacade::get()->changeCell(extendedObject);
    }
}

void _InspectionWindow::applyPendingSignalEntries(ExtendedObjectDesc& extendedObject)
{
    if (!_pendingSignalEntries || extendedObject.object.getObjectType() != ObjectType_Cell) {
        return;
    }

    auto& cell = extendedObject.object.getCellRef();
    if (cell.getCellType() == CellType_Memory) {
        auto& memory = std::get<MemoryDesc>(cell._cellType);
        memory._signalEntries = *_pendingSignalEntries;
    }
    _pendingSignalEntries.reset();
}

void _InspectionWindow::processParticle(EnergyDesc particle)
{
    auto origParticle = particle;
    AlienGui::DynamicTableLayout table(TableColumnWidth);
    if (table.begin()) {
        auto energyParticleOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Energy particle").rank(AlienGui::TreeNodeRank::High));
        if (energyParticleOpen) {
            processPropertiesSubNode("Energy particle", [&] {
                inspectorHexId("Particle id", particle._id);
                AlienGui::InputFloat2(AlienGui::InputFloat2Parameters().name("Position").format("%.3f").textWidth(TextWidth), particle._pos.x, particle._pos.y);
                AlienGui::InputFloat2(AlienGui::InputFloat2Parameters().name("Velocity").format("%.3f").textWidth(TextWidth), particle._vel.x, particle._vel.y);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(TextWidth), particle._energy);
                AlienGui::ComboColor(
                    AlienGui::ComboColorParameters()
                        .customizationColors(_SimulationFacade::get()->getSimulationParameters().customizationColors.value)
                        .name("Color")
                        .textWidth(TextWidth),
                    particle._color);
            });
        }
        AlienGui::EndTreeNode();
        table.next();
        table.end();
    }
    if (particle != origParticle) {
        _SimulationFacade::get()->changeParticle(particle);
    }
}

void _InspectionWindow::processObjectNode(ObjectDesc& object)
{
    auto objectOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Object").rank(AlienGui::TreeNodeRank::High));
    if (objectOpen) {
        processPropertiesSubNode("Object", [&] {
            inspectorHexId("Object id", object._id);
            AlienGui::InputFloat2(AlienGui::InputFloat2Parameters().name("Position").format("%.2f").textWidth(TextWidth), object._pos.x, object._pos.y);
            AlienGui::InputFloat2(AlienGui::InputFloat2Parameters().name("Velocity").format("%.2f").textWidth(TextWidth), object._vel.x, object._vel.y);
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Stiffness").format("%.2f").step(0.05f).textWidth(TextWidth), object._stiffness);
            AlienGui::ComboColor(
                AlienGui::ComboColorParameters()
                    .customizationColors(_SimulationFacade::get()->getSimulationParameters().customizationColors.value)
                    .name("Color")
                    .textWidth(TextWidth),
                object._color);
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Fixed").textWidth(TextWidth), object._fixed);
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Sticky").textWidth(TextWidth), object._sticky);

            auto objectType = object.getObjectType();
            AlienGui::ComboParameters typeParams;
            typeParams.name("Object type").textWidth(TextWidth).values(Const::ObjectTypeStrings);
            if (AlienGui::Combo(typeParams, objectType)) {
                object._type = createObjectTypeDesc(objectType);
            }
        });

        if (!object._connections.empty()) {
            auto connectionsOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Connections").rank(AlienGui::TreeNodeRank::Default));
            if (connectionsOpen) {
                for (size_t i = 0; i < object._connections.size(); ++i) {
                    auto& conn = object._connections.at(i);
                    auto const connectionNumber = i + 1;
                    if (AlienGui::BeginTreeNode(
                            AlienGui::TreeNodeParameters().name("Connection #" + std::to_string(connectionNumber)).rank(AlienGui::TreeNodeRank::Low))) {
                        inspectorHexId("Connected id", conn._objectId);
                        AlienGui::InputFloat(
                            AlienGui::InputFloatParameters().name("Distance").format("%.2f").textWidth(TextWidth).readOnly(true), conn._distance);
                        AlienGui::InputFloat(
                            AlienGui::InputFloatParameters().name("Ref angle").format("%.2f").textWidth(TextWidth).readOnly(true), conn._angleFromPrevious);
                    }
                    AlienGui::EndTreeNode();
                }
            }
            AlienGui::EndTreeNode();
        }
    }
    AlienGui::EndTreeNode();
}

void _InspectionWindow::processSolidNode(ObjectDesc& object)
{
    auto solidOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Solid").rank(AlienGui::TreeNodeRank::High).defaultOpen(false));
    if (solidOpen) {
        processPropertiesSubNode("Solid", [&] {
            auto& solid = object.getSolidRef();
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(TextWidth), solid._energy);
        });
    }
    AlienGui::EndTreeNode();
}

void _InspectionWindow::processFluidNode(ObjectDesc& object)
{
    auto fluidOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Fluid").rank(AlienGui::TreeNodeRank::High).defaultOpen(false));
    if (fluidOpen) {
        processPropertiesSubNode("Fluid", [&] {
            auto& fluid = object.getFluidRef();
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(TextWidth), fluid._energy);
            AlienGui::SliderFloat(AlienGui::SliderFloatParameters().name("Glow").min(0.0f).max(1.0f).format("%.2f").textWidth(TextWidth), &fluid._glow);
        });
    }
    AlienGui::EndTreeNode();
}

void _InspectionWindow::processFreeCellNode(ObjectDesc& object)
{
    auto freeCellOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Free cell").rank(AlienGui::TreeNodeRank::High).defaultOpen(false));
    if (freeCellOpen) {
        processPropertiesSubNode("Free cell", [&] {
            auto& freeCell = object.getFreeCellRef();
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Energy").format("%.2f").textWidth(TextWidth), freeCell._energy);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Age").textWidth(TextWidth), freeCell._age);
        });
    }
    AlienGui::EndTreeNode();
}

void _InspectionWindow::processCellNode(ObjectDesc& object)
{
    auto cellOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell").rank(AlienGui::TreeNodeRank::High).defaultOpen(false));
    if (cellOpen) {
        auto& cell = object.getCellRef();
        processPropertiesSubNode("Cell", [&] {
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Usable energy").format("%.2f").textWidth(TextWidth), cell._usableEnergy);
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Raw energy").format("%.2f").textWidth(TextWidth), cell._rawEnergy);
            AlienGui::InputOptionalFloat(AlienGui::InputFloatParameters().name("Front angle").format("%.2f").textWidth(TextWidth), cell._frontAngle);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Age").textWidth(TextWidth), cell._age);
            static std::vector<std::string> const cellStateStrings = {"Ready", "Constructing", "Activating", "Dying", "Instant dying"};
            AlienGui::ComboParameters stateParams;
            stateParams.name("Cell state").textWidth(TextWidth).values(cellStateStrings);
            AlienGui::Combo(stateParams, cell._cellState);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Node index").textWidth(TextWidth), cell._nodeIndex);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Parent node index").textWidth(TextWidth), cell._parentNodeIndex);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Gene index").textWidth(TextWidth), cell._geneIndex);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Concatenation index").textWidth(TextWidth), cell._concatenationIndex);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Branch index").textWidth(TextWidth), cell._branchIndex);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Activation time").textWidth(TextWidth), cell._activationTime);
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Head cell").textWidth(TextWidth), cell._headCell);

            auto cellType = cell.getCellType();
            AlienGui::ComboParameters cellTypeParams;
            cellTypeParams.name("Cell type").textWidth(TextWidth).values(Const::CellTypeStrings);
            if (AlienGui::Combo(cellTypeParams, cellType)) {
                cell._cellType = createCellTypeDesc(cellType);
            }
        });

        processCellTypeNode(cell);

        if (cell._constructor.has_value()) {
            processConstructorNode(cell._constructor.value());
        }

        processSignalsNode(cell);
        processNeuralNetNode(cell);
    }
    AlienGui::EndTreeNode();
}

void _InspectionWindow::processConstructorNode(ConstructorDesc& constructor)
{
    auto constructorOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Constructor").rank(AlienGui::TreeNodeRank::Default).defaultOpen(false));
    if (constructorOpen) {
        AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Auto trigger interval").textWidth(TextWidth), constructor._autoTriggerInterval);
        AlienGui::InputInt(AlienGui::InputIntParameters().name("Activation time").textWidth(TextWidth), constructor._constructionActivationTime);
        AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Construction angle").format("%.2f").textWidth(TextWidth), constructor._constructionAngle);
        static std::vector<std::string> const provideEnergyStrings = {"Cell only", "Free generation"};
        AlienGui::ComboParameters provideParams;
        provideParams.name("Provide energy").textWidth(TextWidth).values(provideEnergyStrings);
        int providedEnergy = constructor._provideEnergy;
        if (AlienGui::Combo(provideParams, providedEnergy)) {
            constructor._provideEnergy = static_cast<ProvideEnergy>(providedEnergy);
        }
        AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Reserved energy").format("%.2f").textWidth(TextWidth), constructor._reservedEnergy);
        AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Separation").textWidth(TextWidth), constructor._separation);
        if (!constructor._separation) {
            auto numBranches = constructor._numBranches - 1;
            AlienGui::Switcher(
                AlienGui::SwitcherParameters().name("Number of branches").values({"1", "2", "3", "4", "5", "6"}).textWidth(TextWidth), numBranches);
            constructor._numBranches = numBranches + 1;
        }
        AlienGui::InputInt(AlienGui::InputIntParameters().name("Concatenations").infinity(true).textWidth(TextWidth), constructor._numConcatenations);
        AlienGui::InputInt(AlienGui::InputIntParameters().name("Gene index").textWidth(TextWidth), constructor._geneIndex);
    }
    AlienGui::EndTreeNode();
}


void _InspectionWindow::processCreatureNode(ExtendedObjectDesc& extendedObject)
{
    auto creatureOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Creature").rank(AlienGui::TreeNodeRank::High).defaultOpen(false));
    if (creatureOpen) {
        processPropertiesSubNode("Creature", [&] {
            auto& creature = extendedObject.creature.value();
            inspectorHexId("Creature id", creature._id);
            inspectorText("Generation", std::to_string(creature._generation));
            inspectorText("Num cells", std::to_string(creature._numCells));
            auto& genome = extendedObject.genome.value();
            std::stringstream frontAngle;
            frontAngle << std::fixed << std::setprecision(1) << genome._frontAngle;
            inspectorText("Front angle", frontAngle.str());
            inspectorText("Genome name", genome._name);
            inspectorText("Lineage id", std::to_string(genome._lineageId));
            if (AlienGui::Button(AlienGui::ButtonParameters().buttonText("Edit").name("Edit genome").textWidth(TextWidth))) {
                GenomeEditorWindow::get().openTab(genome, false);
            }
        });
    }
    AlienGui::EndTreeNode();
}

void _InspectionWindow::processSignalsNode(CellDesc& cell)
{
    auto signalsOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Signals").rank(AlienGui::TreeNodeRank::Default).defaultOpen(false));
    if (signalsOpen) {
        auto& channels = cell._signal._channels;
        if (static_cast<int>(channels.size()) < NEURONS_PER_CELL) {
            channels.resize(NEURONS_PER_CELL, 0.0f);
        }
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("#" + std::to_string(i + 1)).min(-2.0f).max(2.0f).format("%.2f").textWidth(TextWidth), &channels.at(i));
        }
    }
    AlienGui::EndTreeNode();
}

void _InspectionWindow::processNeuralNetNode(CellDesc& cell)
{
    auto neuralNetOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Neural network").rank(AlienGui::TreeNodeRank::Default).defaultOpen(false));
    if (neuralNetOpen) {
        _neuralNetWidget->process(
            cell._neuralNetwork._weights, cell._neuralNetwork._biases, cell._neuralNetwork._activationFunctions, cell._neuralNetwork._connectionWeights);
    }
    AlienGui::EndTreeNode();
}

namespace
{
    void processMemoryChannelBits(uint16_t& bitMask)
    {
        bool bits[NEURONS_PER_CELL];
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            bits[i] = (bitMask & (1 << i)) != 0;
        }
        AlienGui::MultiCheckboxes(AlienGui::MultiCheckboxesParameters().name("Channel mask bit 0-3").textWidth(TextWidth), bits[0], bits[1], bits[2], bits[3]);
        AlienGui::MultiCheckboxes(AlienGui::MultiCheckboxesParameters().name("Channel mask bit 4-7").textWidth(TextWidth), bits[4], bits[5], bits[6], bits[7]);
        AlienGui::MultiCheckboxes(
            AlienGui::MultiCheckboxesParameters().name("Channel mask bit 8-11").textWidth(TextWidth), bits[8], bits[9], bits[10], bits[11]);
        AlienGui::MultiCheckboxes(
            AlienGui::MultiCheckboxesParameters().name("Channel mask bit 12-15").textWidth(TextWidth), bits[12], bits[13], bits[14], bits[15]);
        bitMask = 0;
        for (int i = 0; i < NEURONS_PER_CELL; ++i) {
            if (bits[i]) {
                bitMask |= 1 << i;
            }
        }
    }
}

void _InspectionWindow::processCellTypeNode(CellDesc& cell)
{
    auto cellTypeOpen = AlienGui::BeginTreeNode(AlienGui::TreeNodeParameters().name("Cell type##Cell node").rank(AlienGui::TreeNodeRank::Default).defaultOpen(false));
    if (cellTypeOpen) {
        auto cellType = cell.getCellType();
        auto const& customizationColors = _SimulationFacade::get()->getSimulationParameters().customizationColors.value;

        if (cellType == CellType_Depot) {
            auto& depot = std::get<DepotDesc>(cell._cellType);
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Storage limit").format("%.2f").textWidth(TextWidth), depot._storageLimit);
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Stored usable energy").format("%.2f").textWidth(TextWidth), depot._storedUsableEnergy);
        } else if (cellType == CellType_Sensor) {
            auto& sensor = std::get<SensorDesc>(cell._cellType);
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Auto trigger").textWidth(TextWidth), sensor._autoTrigger);
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Tag for attackers").textWidth(TextWidth), sensor._tagForAttackers);
            auto mode = sensor.getMode();
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::SensorModeStrings);
            if (AlienGui::Combo(modeParams, mode)) {
                sensor._mode = createSensorModeDesc(mode);
            }
            if (mode == SensorMode_DetectEnergy) {
                auto& m = std::get<DetectEnergyDesc>(sensor._mode);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Min density").step(0.05f).format("%.2f").textWidth(TextWidth), m._minDensity);
            } else if (mode == SensorMode_DetectFreeCell) {
                auto& m = std::get<DetectFreeCellDesc>(sensor._mode);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Min density").step(0.05f).format("%.2f").textWidth(TextWidth), m._minDensity);
                AlienGui::ColorCheckboxes(
                    AlienGui::ColorCheckboxesParameters().customizationColors(customizationColors).name("Restrict to colors").textWidth(TextWidth),
                    m._restrictToColors);
            } else if (mode == SensorMode_DetectCreature) {
                auto& m = std::get<DetectCreatureDesc>(sensor._mode);
                AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Min num cells").textWidth(TextWidth), m._minNumCells);
                AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Max num cells").textWidth(TextWidth), m._maxNumCells);
                AlienGui::ColorCheckboxes(
                    AlienGui::ColorCheckboxesParameters().customizationColors(customizationColors).name("Restrict to colors").textWidth(TextWidth),
                    m._restrictToColors);
                AlienGui::ComboParameters lineageParams;
                lineageParams.name("Restrict to lineage").textWidth(TextWidth).values({"No", "Same lineage", "Other lineage"});
                AlienGui::Combo(lineageParams, m._restrictToLineage);
            }
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Min range").textWidth(TextWidth), sensor._minRange);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Max range").textWidth(TextWidth), sensor._maxRange);
        } else if (cellType == CellType_Generator) {
            auto& generator = std::get<GeneratorDesc>(cell._cellType);
            AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Additive").textWidth(TextWidth), generator._additive);
            AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Value offset").step(0.05f).format("%.2f").textWidth(TextWidth), generator._valueOffset);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Time offset").textWidth(TextWidth), generator._timeOffset);
            auto mode = generator.getMode();
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::GeneratorModeStrings);
            if (AlienGui::Combo(modeParams, mode)) {
                generator._mode = createGeneratorModeDesc(mode);
            }
            if (mode == GeneratorMode_SquareSignal) {
                auto& m = std::get<SquareSignalDesc>(generator._mode);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Amplitude").step(0.05f).format("%.2f").textWidth(TextWidth), m._amplitude);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Period").textWidth(TextWidth), m._period);
            } else if (mode == GeneratorMode_SawtoothSignal) {
                auto& m = std::get<SawtoothSignalDesc>(generator._mode);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Amplitude").step(0.05f).format("%.2f").textWidth(TextWidth), m._amplitude);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Period").textWidth(TextWidth), m._period);
            }
        } else if (cellType == CellType_Attacker) {
            auto& attacker = std::get<AttackerDesc>(cell._cellType);
            auto mode = attacker.getMode();
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::AttackerModeStrings);
            if (AlienGui::Combo(modeParams, mode)) {
                attacker._mode = createAttackerModeDesc(mode);
            }
            if (mode == AttackerMode_FreeCell) {
                auto& m = std::get<AttackFreeCellDesc>(attacker._mode);
                AlienGui::ColorCheckboxes(
                    AlienGui::ColorCheckboxesParameters().customizationColors(customizationColors).name("Restrict to colors").textWidth(TextWidth),
                    m._restrictToColors);
            }
        } else if (cellType == CellType_Injector) {
            auto& injector = std::get<InjectorDesc>(cell._cellType);
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Gene index").textWidth(TextWidth), injector._geneIndex);
        } else if (cellType == CellType_Muscle) {
            auto& muscle = std::get<MuscleDesc>(cell._cellType);
            auto mode = muscle.getMode();
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::MuscleModeStrings);
            if (AlienGui::Combo(modeParams, mode)) {
                muscle._mode = createMuscleModeDesc(mode);
            }
            if (mode == MuscleMode_AutoBending) {
                auto& m = std::get<AutoBendingDesc>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max angle deviation").step(0.05f).format("%.2f").textWidth(TextWidth), m._maxAngleDeviation);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Forward backward ratio").step(0.05f).format("%.2f").textWidth(TextWidth), m._forwardBackwardRatio);
            } else if (mode == MuscleMode_ManualBending) {
                auto& m = std::get<ManualBendingDesc>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max angle deviation").step(0.05f).format("%.2f").textWidth(TextWidth), m._maxAngleDeviation);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Forward backward ratio").step(0.05f).format("%.2f").textWidth(TextWidth), m._forwardBackwardRatio);
            } else if (mode == MuscleMode_AngleBending) {
                auto& m = std::get<AngleBendingDesc>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max angle deviation").step(0.05f).format("%.2f").textWidth(TextWidth), m._maxAngleDeviation);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Attraction repulsion ratio").step(0.05f).format("%.2f").textWidth(TextWidth),
                    m._attractionRepulsionRatio);
            } else if (mode == MuscleMode_AutoCrawling) {
                auto& m = std::get<AutoCrawlingDesc>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max distance deviation").step(0.01f).format("%.2f").textWidth(TextWidth), m._maxDistanceDeviation);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Forward backward ratio").step(0.05f).format("%.2f").textWidth(TextWidth), m._forwardBackwardRatio);
            } else if (mode == MuscleMode_ManualCrawling) {
                auto& m = std::get<ManualCrawlingDesc>(muscle._mode);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Max distance deviation").step(0.01f).format("%.2f").textWidth(TextWidth), m._maxDistanceDeviation);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Forward backward ratio").step(0.05f).format("%.2f").textWidth(TextWidth), m._forwardBackwardRatio);
            }
        } else if (cellType == CellType_Defender) {
            auto& defender = std::get<DefenderDesc>(cell._cellType);
            int defMode = defender._mode;
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::DefenderModeStrings);
            if (AlienGui::Combo(modeParams, defMode)) {
                defender._mode = static_cast<DefenderMode>(defMode);
            }
        } else if (cellType == CellType_Reconnector) {
            auto& reconnector = std::get<ReconnectorDesc>(cell._cellType);
            auto mode = reconnector.getMode();
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::ReconnectorModeStrings);
            if (AlienGui::Combo(modeParams, mode)) {
                reconnector._mode = createReconnectorModeDesc(mode);
            }
            if (mode == ReconnectorMode_FreeCell) {
                auto& m = std::get<ReconnectFreeCellDesc>(reconnector._mode);
                AlienGui::ColorCheckboxes(
                    AlienGui::ColorCheckboxesParameters().customizationColors(customizationColors).name("Restrict to colors").textWidth(TextWidth),
                    m._restrictToColors);
            } else if (mode == ReconnectorMode_Creature) {
                auto& m = std::get<ReconnectCreatureDesc>(reconnector._mode);
                AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Min num cells").textWidth(TextWidth), m._minNumCells);
                AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Max num cells").textWidth(TextWidth), m._maxNumCells);
                AlienGui::ColorCheckboxes(
                    AlienGui::ColorCheckboxesParameters().customizationColors(customizationColors).name("Restrict to colors").textWidth(TextWidth),
                    m._restrictToColors);
                AlienGui::ComboParameters lineageParams;
                lineageParams.name("Restrict to lineage").textWidth(TextWidth).values({"No", "Same lineage", "Other lineage"});
                AlienGui::Combo(lineageParams, m._restrictToLineage);
            }
        } else if (cellType == CellType_Detonator) {
            auto& detonator = std::get<DetonatorDesc>(cell._cellType);
            static std::vector<std::string> const detonatorStateStrings = {"Ready", "Activated", "Exploded"};
            int state = detonator._state;
            AlienGui::ComboParameters stateParams;
            stateParams.name("State").textWidth(TextWidth).values(detonatorStateStrings);
            if (AlienGui::Combo(stateParams, state)) {
                detonator._state = static_cast<DetonatorState>(state);
            }
            AlienGui::InputInt(AlienGui::InputIntParameters().name("Countdown").textWidth(TextWidth), detonator._countdown);
        } else if (cellType == CellType_Digestor) {
            auto& digestor = std::get<DigestorDesc>(cell._cellType);
            AlienGui::SliderFloat(
                AlienGui::SliderFloatParameters().name("Raw energy conductivity").max(1.0f).format("%.2f").textWidth(TextWidth),
                &digestor._rawEnergyConductivity);
            auto conversion = digestor.getRawEnergyConversionRate();
            AlienGui::SliderFloat(AlienGui::SliderFloatParameters().name("Energy conversion").max(1.0f).format("%.2f").textWidth(TextWidth), &conversion);
            digestor.setRawEnergyConversionRate(conversion);
        } else if (cellType == CellType_Memory) {
            auto& memory = std::get<MemoryDesc>(cell._cellType);
            auto mode = memory.getMode();
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::MemoryModeStrings);
            if (AlienGui::Combo(modeParams, mode)) {
                memory._mode = createMemoryModeDesc(mode);
                if (mode == MemoryMode_SignalRecorder || mode == MemoryMode_SignalStorage) {
                    memory._signalEntries.resize(8, SignalEntryDesc());
                }
            }
            if (mode == MemoryMode_SignalDelay) {
                auto& m = std::get<SignalDelayDesc>(memory._mode);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Delay").textWidth(TextWidth), m._delay);
            } else if (mode == MemoryMode_SignalRecorder) {
                auto& m = std::get<SignalRecorderDesc>(memory._mode);
                AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Read only").textWidth(TextWidth), m._readOnly);
            } else if (mode == MemoryMode_SignalStorage) {
                auto& m = std::get<SignalStorageDesc>(memory._mode);
                AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Read only").textWidth(TextWidth), m._readOnly);
            } else if (mode == MemoryMode_SignalIntegrator) {
                auto& m = std::get<SignalIntegratorDesc>(memory._mode);
                AlienGui::SliderFloat(
                    AlienGui::SliderFloatParameters().name("New signal weight").max(1.0f).format("%.2f").textWidth(TextWidth), &m._newSignalWeight);
            }
            processMemoryChannelBits(memory._channelBitMask);
            if (AlienGui::Button(AlienGui::ButtonParameters().buttonText("Edit").name("Signal buffer").textWidth(TextWidth))) {
                SignalsBufferDialog::get().open(
                    memory._signalEntries, [this](std::vector<SignalEntryDesc> const& entries) { _pendingSignalEntries = entries; });
            }
        } else if (cellType == CellType_Communicator) {
            auto& communicator = std::get<CommunicatorDesc>(cell._cellType);
            auto mode = communicator.getMode();
            AlienGui::ComboParameters modeParams;
            modeParams.name("Mode").textWidth(TextWidth).values(Const::CommunicatorModeStrings);
            if (AlienGui::Combo(modeParams, mode)) {
                communicator._mode = createCommunicatorModeDesc(mode);
            }
            if (mode == CommunicatorMode_Sender) {
                auto& m = std::get<SenderDesc>(communicator._mode);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Range").textWidth(TextWidth), m._range);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Max times sent").textWidth(TextWidth), m._maxTimesSent);
            } else if (mode == CommunicatorMode_Receiver) {
                auto& m = std::get<ReceiverDesc>(communicator._mode);
                AlienGui::ColorCheckboxes(
                    AlienGui::ColorCheckboxesParameters().customizationColors(customizationColors).name("Restrict to colors").textWidth(TextWidth),
                    m._restrictToColors);
                AlienGui::ComboParameters lineageParams;
                lineageParams.name("Restrict to lineage").textWidth(TextWidth).values({"No", "Same lineage", "Other lineage"});
                AlienGui::Combo(lineageParams, m._restrictToLineage);
            }
        }
    }
    AlienGui::EndTreeNode();
}

float _InspectionWindow::calcWindowWidth() const
{
    if (isObject()) {
        return StyleRepository::get().scale(CellWindowWidth);
    }
    return StyleRepository::get().scale(ParticleWindowWidth);
}

void _InspectionWindow::validateAndCorrect(ObjectDesc& object) const
{
    object._stiffness = std::clamp(object._stiffness, 0.0f, 1.0f);
    if (object.getObjectType() == ObjectType_Cell) {
        auto& cell = object.getCellRef();
        cell._usableEnergy = std::max(0.0f, cell._usableEnergy);
        cell._rawEnergy = std::max(0.0f, cell._rawEnergy);
        cell._age = std::max(0, cell._age);
        if (cell._constructor.has_value()) {
            auto& c = cell._constructor.value();
            if (c._autoTriggerInterval.has_value()) {
                c._autoTriggerInterval = std::max(0, c._autoTriggerInterval.value());
            }
            c._reservedEnergy = std::max(0.0f, c._reservedEnergy);
            c._numBranches = std::clamp(c._numBranches, 1, 6);
            c._numConcatenations = std::max(c._numConcatenations, 1);
        }
        if (cell.getCellType() == CellType_Sensor) {
            auto& sensor = std::get<SensorDesc>(cell._cellType);
            sensor._minRange = std::clamp(sensor._minRange, 0, 255);
            sensor._maxRange = std::clamp(sensor._maxRange, 0, 255);
        }
        if (cell.getCellType() == CellType_Generator) {
            auto& generator = std::get<GeneratorDesc>(cell._cellType);
            if (auto* sq = std::get_if<SquareSignalDesc>(&generator._mode)) {
                sq->_period = std::max(1, sq->_period);
            } else if (auto* sw = std::get_if<SawtoothSignalDesc>(&generator._mode)) {
                sw->_period = std::max(1, sw->_period);
            }
        }
        if (cell.getCellType() == CellType_Detonator) {
            auto& detonator = std::get<DetonatorDesc>(cell._cellType);
            detonator._countdown = std::clamp(detonator._countdown, 0, 0xffff);
        }
    }
}
