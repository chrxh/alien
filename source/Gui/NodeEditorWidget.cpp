#include "NodeEditorWidget.h"

#include <ranges>

#include <boost/range/adaptors.hpp>

#include "AlienGui.h"
#include "GenomeTabEditData.h"
#include "GenomeTabLayoutData.h"
#include "LoginDialog.h"
#include "NeuralNetEditorWidget.h"

namespace
{
    auto constexpr MinTextWidth = 150.0f;
    auto constexpr MaxWidgetWidth = 200.0f;
    auto constexpr MinColumnWidth = 300.0f;
}


NodeEditorWidget _NodeEditorWidget::create(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
{
    return NodeEditorWidget(new _NodeEditorWidget(editData, layoutData));
}

void _NodeEditorWidget::process()
{
    if (ImGui::BeginChild("NodeEditor", ImVec2(0, 0))) {
        auto nodeIndex = _editData->getSelectedNodeIndex();
        if (nodeIndex.has_value()) {
            ImGui::PushID(_editData->selectedGeneIndex.value());
            ImGui::PushID(nodeIndex.value());
            processNodeAttributes();

            AlienGui::MovableHorizontalSeparator(AlienGui::MovableHorizontalSeparatorParameters().additive(false), _layoutData->neuralNetEditorHeight);

            processNeuralNetEditor();
            ImGui::PopID();
            ImGui::PopID();
        } else {
            processNoSelection();
        }
    }
    ImGui::EndChild();
}

_NodeEditorWidget::_NodeEditorWidget(GenomeTabEditData const& editData, GenomeTabLayoutData const& layoutData)
    : _editData(editData)
    , _layoutData(layoutData)
{
    _neuralNetWidget = _NeuralNetEditorWidget::create();
}

namespace
{
    CellTypeGenomeDesc createCellTypeGenomeDesc(CellType cellType)
    {
        switch (cellType) {
        case CellType_Base:
            return BaseGenomeDesc();
        case CellType_Depot:
            return DepotGenomeDesc();
        case CellType_Sensor:
            return SensorGenomeDesc();
        case CellType_Generator:
            return GeneratorGenomeDesc();
        case CellType_Attacker:
            return AttackerGenomeDesc();
        case CellType_Injector:
            return InjectorGenomeDesc();
        case CellType_Muscle:
            return MuscleGenomeDesc();
        case CellType_Defender:
            return DefenderGenomeDesc();
        case CellType_Reconnector:
            return ReconnectorGenomeDesc();
        case CellType_Detonator:
            return DetonatorGenomeDesc();
        case CellType_Digestor:
            return DigestorGenomeDesc();
        case CellType_Memory:
            return MemoryGenomeDesc();
        case CellType_Communicator:
            return CommunicatorGenomeDesc();
        default:
            CHECK(false);
        }
    }

    MuscleModeGenomeDesc createMuscleModeGenomeDesc(MuscleMode mode)
    {
        switch (mode) {
        case MuscleMode_AutoBending:
            return AutoBendingGenomeDesc();
        case MuscleMode_ManualBending:
            return ManualBendingGenomeDesc();
        case MuscleMode_AngleBending:
            return AngleBendingGenomeDesc();
        case MuscleMode_AutoCrawling:
            return AutoCrawlingGenomeDesc();
        case MuscleMode_ManualCrawling:
            return ManualCrawlingGenomeDesc();
        case MuscleMode_DirectMovement:
            return DirectMovementGenomeDesc();
        default:
            CHECK(false);
        }
    }

    SensorModeGenomeDesc createSensorModeGenomeDesc(SensorMode mode)
    {
        switch (mode) {
        case SensorMode_Telemetry:
            return TelemetryGenomeDesc();
        case SensorMode_DetectEnergy:
            return DetectEnergyGenomeDesc();
        case SensorMode_DetectStructure:
            return DetectStructureGenomeDesc();
        case SensorMode_DetectFreeCell:
            return DetectFreeCellGenomeDesc();
        case SensorMode_DetectCreature:
            return DetectCreatureGenomeDesc();
        default:
            CHECK(false);
        }
    }

    AttackerModeGenomeDesc createAttackerModeGenomeDesc(AttackerMode mode)
    {
        switch (mode) {
        case AttackerMode_FreeCell:
            return AttackFreeCellGenomeDesc();
        case AttackerMode_Creature:
            return AttackCreatureGenomeDesc();
        default:
            CHECK(false);
        }
    }

    GeneratorModeGenomeDesc createGeneratorModeGenomeDesc(GeneratorMode mode)
    {
        switch (mode) {
        case GeneratorMode_SquareSignal:
            return SquareSignalGenomeDesc();
        case GeneratorMode_SawtoothSignal:
            return SawtoothSignalGenomeDesc();
        default:
            CHECK(false);
        }
    }

    ReconnectorModeGenomeDesc createReconnectorModeGenomeDesc(ReconnectorMode mode)
    {
        switch (mode) {
        case ReconnectorMode_Structure:
            return ReconnectStructureGenomeDesc();
        case ReconnectorMode_FreeCell:
            return ReconnectFreeCellGenomeDesc();
        case ReconnectorMode_Creature:
            return ReconnectCreatureGenomeDesc();
        default:
            CHECK(false);
        }
    }

    MemoryModeGenomeDesc createMemoryModeGenomeDesc(MemoryMode mode)
    {
        switch (mode) {
        case MemoryMode_SignalDelay:
            return SignalDelayGenomeDesc();
        case MemoryMode_SignalRecorder:
            return SignalRecorderGenomeDesc();
        case MemoryMode_SignalStorage:
            return SignalStorageGenomeDesc();
        case MemoryMode_SignalIntegrator:
            return SignalIntegratorGenomeDesc();
        default:
            CHECK(false);
        }
    }

    CommunicatorModeGenomeDesc createCommunicatorModeGenomeDesc(CommunicatorMode mode)
    {
        switch (mode) {
        case CommunicatorMode_Sender:
            return SenderGenomeDesc();
        case CommunicatorMode_Receiver:
            return ReceiverGenomeDesc();
        default:
            CHECK(false);
        }
    }
}

void _NodeEditorWidget::processNodeAttributes()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Selected node").highlighted(true));

    if (ImGui::BeginChild("NodeData", ImVec2(0, -_layoutData->neuralNetEditorHeight), 0, 0)) {
        auto& gene = _editData->getSelectedGeneRef();
        auto& node = _editData->getSelectedNodeRef();

        AlienGui::DynamicTableLayout table(MinColumnWidth);
        if (table.begin()) {
            auto rightColumnWidth = std::max(MinTextWidth, scaleInverse(ImGui::GetContentRegionAvail().x - scale(MaxWidgetWidth)));

            // Angle
            AlienGui::Group(AlienGui::GroupParameters().text("Base properties"));

            auto nodeIndex = _editData->getSelectedNodeIndex();
            auto isInnerNode = nodeIndex.value() != 0 && nodeIndex != gene._nodes.size() - 1;
            if (AlienGui::InputFloat(
                    AlienGui::InputFloatParameters()
                        .name("Angle")
                        .textWidth(rightColumnWidth)
                        .infoLabel(gene._shape != ConstructorShape_Custom && isInnerNode ? std::make_optional(std::string("Deduced")) : std::nullopt)
                        .format("%.1f"),
                    node._referenceAngle)) {
                if (isInnerNode) {
                    gene._shape = ConstructorShape_Custom;
                }
            }

            // Previous nodes connections
            if (nodeIndex != 0) {
                auto numAdditionalConnections = node._numAdditionalConnections + 1;
                if (AlienGui::InputInt(
                        AlienGui::InputIntParameters()
                            .name("Prev nodes connections")
                            .infoLabel(gene._shape != ConstructorShape_Custom ? std::make_optional(std::string("Deduced")) : std::nullopt)
                            .textWidth(rightColumnWidth),
                        numAdditionalConnections)) {
                    gene._shape = ConstructorShape_Custom;
                }
                node._numAdditionalConnections = std::max(numAdditionalConnections - 1, 0);
            }

            AlienGui::ComboColor(AlienGui::ComboColorParameters().name("Color").textWidth(rightColumnWidth), node._color);

            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Type-specific properties"));

            // Type
            auto nodeType = node.getCellType();
            if (AlienGui::Combo(AlienGui::ComboParameters().name("Type").values(Const::CellTypeStrings).textWidth(rightColumnWidth), nodeType)) {
                node._cellType = createCellTypeGenomeDesc(nodeType);
            }

            if (nodeType == CellType_Base) {
            } else if (nodeType == CellType_Depot) {
                AlienGui::BeginIndent();
                auto& depot = std::get<DepotGenomeDesc>(node._cellType);
                AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Max energy for storage").textWidth(rightColumnWidth), depot._storageLimit);
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Initial stored energy").textWidth(rightColumnWidth), depot._initialStoredUsableEnergy);
                AlienGui::EndIndent();
            } else if (nodeType == CellType_Sensor) {

                ImGui::PushID("Sensor");
                AlienGui::BeginIndent();

                // Auto activation interval
                auto& sensor = std::get<SensorGenomeDesc>(node._cellType);
                AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Auto trigger").textWidth(rightColumnWidth), sensor._autoTrigger);

                // Mode selection
                auto mode = sensor.getMode();
                if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::SensorModeStrings).textWidth(rightColumnWidth), mode)) {
                    sensor._mode = createSensorModeGenomeDesc(mode);
                }

                // Mode-specific parameters
                if (mode == SensorMode_DetectStructure) {
                    // No parameters
                } else if (mode == SensorMode_DetectEnergy) {
                    AlienGui::BeginIndent();
                    auto& detectEnergy = std::get<DetectEnergyGenomeDesc>(sensor._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Min density").step(0.05f).format("%.2f").textWidth(rightColumnWidth), detectEnergy._minDensity);
                    AlienGui::EndIndent();
                } else if (mode == SensorMode_DetectStructure) {
                    // No parameters
                } else if (mode == SensorMode_DetectFreeCell) {
                    AlienGui::BeginIndent();
                    auto& detectFreeCell = std::get<DetectFreeCellGenomeDesc>(sensor._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Min density").step(0.05f).format("%.2f").textWidth(rightColumnWidth),
                        detectFreeCell._minDensity);
                    AlienGui::ComboOptionalColor(
                        AlienGui::ComboColorParameters().name("Restrict to color").textWidth(rightColumnWidth), detectFreeCell._restrictToColor);
                    AlienGui::EndIndent();
                } else if (mode == SensorMode_DetectCreature) {
                    AlienGui::BeginIndent();
                    auto& detectCreature = std::get<DetectCreatureGenomeDesc>(sensor._mode);
                    AlienGui::InputOptionalInt(
                        AlienGui::InputIntParameters().name("Min creature cells").textWidth(rightColumnWidth), detectCreature._minNumCells);
                    AlienGui::InputOptionalInt(
                        AlienGui::InputIntParameters().name("Max creature cells").textWidth(rightColumnWidth), detectCreature._maxNumCells);
                    AlienGui::ComboOptionalColor(
                        AlienGui::ComboColorParameters().name("Restrict to color").textWidth(rightColumnWidth), detectCreature._restrictToColor);
                    AlienGui::Combo(
                        AlienGui::ComboParameters().name("Restrict to lineage").values({"No", "Same lineage", "Other lineage"}).textWidth(rightColumnWidth),
                        detectCreature._restrictToLineage);
                    AlienGui::EndIndent();
                }

                // Minimum range
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Min range").textWidth(rightColumnWidth), sensor._minRange);

                // Maximum range
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Max range").textWidth(rightColumnWidth), sensor._maxRange);

                AlienGui::EndIndent();
                ImGui::PopID();

            } else if (nodeType == CellType_Generator) {

                AlienGui::BeginIndent();

                auto& generator = std::get<GeneratorGenomeDesc>(node._cellType);

                // Additive
                AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Additive").textWidth(rightColumnWidth), generator._additive);

                // Value offset
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Value offset").format("%.2f").step(0.05f).textWidth(rightColumnWidth), generator._valueOffset);

                // Time offset
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Time offset").textWidth(rightColumnWidth), generator._timeOffset);

                // Mode
                auto mode = generator.getMode();
                if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::GeneratorModeStrings).textWidth(rightColumnWidth), mode)) {
                    generator._mode = createGeneratorModeGenomeDesc(mode);
                }

                if (mode == GeneratorMode_SquareSignal) {
                    AlienGui::BeginIndent();

                    auto& squareSignal = std::get<SquareSignalGenomeDesc>(generator._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Amplitude").format("%.2f").step(0.05f).textWidth(rightColumnWidth), squareSignal._amplitude);
                    AlienGui::InputInt(AlienGui::InputIntParameters().name("Period").textWidth(rightColumnWidth), squareSignal._period);

                    AlienGui::EndIndent();
                } else if (mode == GeneratorMode_SawtoothSignal) {
                    AlienGui::BeginIndent();

                    auto& sawtoothSignal = std::get<SawtoothSignalGenomeDesc>(generator._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Amplitude").format("%.2f").step(0.05f).textWidth(rightColumnWidth), sawtoothSignal._amplitude);
                    AlienGui::InputInt(AlienGui::InputIntParameters().name("Period").textWidth(rightColumnWidth), sawtoothSignal._period);

                    AlienGui::EndIndent();
                }

                AlienGui::EndIndent();

            } else if (nodeType == CellType_Attacker) {

                AlienGui::BeginIndent();

                auto& attacker = std::get<AttackerGenomeDesc>(node._cellType);
                auto mode = attacker.getMode();
                if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::AttackerModeStrings).textWidth(rightColumnWidth), mode)) {
                    attacker._mode = createAttackerModeGenomeDesc(mode);
                }

                if (mode == AttackerMode_FreeCell) {
                    AlienGui::BeginIndent();

                    auto& attackFreeCell = std::get<AttackFreeCellGenomeDesc>(attacker._mode);
                    AlienGui::ComboOptionalColor(
                        AlienGui::ComboColorParameters().name("Restrict to color").textWidth(rightColumnWidth), attackFreeCell._restrictToColor);

                    AlienGui::EndIndent();
                }

                AlienGui::EndIndent();
            } else if (nodeType == CellType_Injector) {

                AlienGui::BeginIndent();

                // Gene index
                auto& injector = std::get<InjectorGenomeDesc>(node._cellType);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Gene index").textWidth(rightColumnWidth), injector._geneIndex);

                AlienGui::EndIndent();

            } else if (nodeType == CellType_Muscle) {

                AlienGui::BeginIndent();

                // Mode
                auto& muscle = std::get<MuscleGenomeDesc>(node._cellType);
                auto mode = muscle.getMode();
                if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::MuscleModeStrings).textWidth(rightColumnWidth), mode)) {
                    muscle._mode = createMuscleModeGenomeDesc(mode);
                }

                if (mode == MuscleMode_AutoBending) {
                    AlienGui::BeginIndent();

                    // Max angle deviation
                    auto& autoBending = std::get<AutoBendingGenomeDesc>(muscle._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        autoBending._maxAngleDeviation);

                    // Front back ratio
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Forward backward ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        autoBending._forwardBackwardRatio);

                    AlienGui::EndIndent();

                } else if (mode == MuscleMode_ManualBending) {
                    AlienGui::BeginIndent();

                    // Max angle deviation
                    auto& manualBending = std::get<ManualBendingGenomeDesc>(muscle._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        manualBending._maxAngleDeviation);

                    // Front back ratio
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Forward backward ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        manualBending._forwardBackwardRatio);

                    AlienGui::EndIndent();

                } else if (mode == MuscleMode_AngleBending) {
                    AlienGui::BeginIndent();

                    // Max angle deviation
                    auto& angleBending = std::get<AngleBendingGenomeDesc>(muscle._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Max angle deviation").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        angleBending._maxAngleDeviation);

                    // Front back ratio
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Attraction repulsion ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        angleBending._attractionRepulsionRatio);

                    AlienGui::EndIndent();

                } else if (mode == MuscleMode_AutoCrawling) {
                    AlienGui::BeginIndent();

                    // Max angle deviation
                    auto& autoCrawling = std::get<AutoCrawlingGenomeDesc>(muscle._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Max distance deviation").format("%.2f").step(0.01f).textWidth(rightColumnWidth),
                        autoCrawling._maxDistanceDeviation);

                    // Front back ratio
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Forward backward ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        autoCrawling._forwardBackwardRatio);

                    AlienGui::EndIndent();

                } else if (mode == MuscleMode_ManualCrawling) {
                    AlienGui::BeginIndent();

                    // Max angle deviation
                    auto& manualCrawling = std::get<ManualCrawlingGenomeDesc>(muscle._mode);
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Max distance deviation").format("%.2f").step(0.01f).textWidth(rightColumnWidth),
                        manualCrawling._maxDistanceDeviation);

                    // Front back ratio
                    AlienGui::InputFloat(
                        AlienGui::InputFloatParameters().name("Forward backward ratio").format("%.2f").step(0.05f).textWidth(rightColumnWidth),
                        manualCrawling._forwardBackwardRatio);

                    AlienGui::EndIndent();

                } else if (mode == MuscleMode_DirectMovement) {
                }

                AlienGui::EndIndent();

            } else if (nodeType == CellType_Defender) {

                AlienGui::BeginIndent();

                // Defender mode
                auto& defender = std::get<DefenderGenomeDesc>(node._cellType);
                AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::DefenderModeStrings).textWidth(rightColumnWidth), defender._mode);

                AlienGui::EndIndent();

            } else if (nodeType == CellType_Reconnector) {

                AlienGui::BeginIndent();

                // Mode selection
                auto& reconnector = std::get<ReconnectorGenomeDesc>(node._cellType);
                auto mode = reconnector.getMode();
                if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::ReconnectorModeStrings).textWidth(rightColumnWidth), mode)) {
                    reconnector._mode = createReconnectorModeGenomeDesc(mode);
                }

                // Mode-specific parameters
                if (mode == ReconnectorMode_Structure) {
                    // No parameters
                } else if (mode == ReconnectorMode_FreeCell) {
                    AlienGui::BeginIndent();
                    auto& freeCell = std::get<ReconnectFreeCellGenomeDesc>(reconnector._mode);
                    AlienGui::ComboOptionalColor(
                        AlienGui::ComboColorParameters().name("Restrict to color").textWidth(rightColumnWidth), freeCell._restrictToColor);
                    AlienGui::EndIndent();
                } else if (mode == ReconnectorMode_Creature) {
                    AlienGui::BeginIndent();
                    auto& creature = std::get<ReconnectCreatureGenomeDesc>(reconnector._mode);
                    AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Min creature cells").textWidth(rightColumnWidth), creature._minNumCells);
                    AlienGui::InputOptionalInt(AlienGui::InputIntParameters().name("Max creature cells").textWidth(rightColumnWidth), creature._maxNumCells);
                    AlienGui::ComboOptionalColor(
                        AlienGui::ComboColorParameters().name("Restrict to color").textWidth(rightColumnWidth), creature._restrictToColor);
                    AlienGui::Combo(
                        AlienGui::ComboParameters().name("Restrict to lineage").values({"No", "Same lineage", "Other lineage"}).textWidth(rightColumnWidth),
                        creature._restrictToLineage);
                    AlienGui::EndIndent();
                }

                AlienGui::EndIndent();

            } else if (nodeType == CellType_Detonator) {

                AlienGui::BeginIndent();

                // Countdown
                auto& detonator = std::get<DetonatorGenomeDesc>(node._cellType);
                AlienGui::InputInt(AlienGui::InputIntParameters().name("Countdown").textWidth(rightColumnWidth), detonator._countdown);

                AlienGui::EndIndent();
            } else if (nodeType == CellType_Digestor) {
                AlienGui::BeginIndent();
                auto& digestor = std::get<DigestorGenomeDesc>(node._cellType);
                AlienGui::SliderFloat(
                    AlienGui::SliderFloatParameters().name("Energy conductivity").max(1.0f).format("%.2f").textWidth(rightColumnWidth),
                    &digestor._rawEnergyConductivity);
                auto rawEnergyConversionRate = digestor.getRawEnergyConversionRate();
                AlienGui::SliderFloat(
                    AlienGui::SliderFloatParameters().name("Energy conversion").max(1.0f).format("%.2f").textWidth(rightColumnWidth), &rawEnergyConversionRate);
                digestor.setRawEnergyConversionRate(rawEnergyConversionRate);
                AlienGui::EndIndent();
            } else if (nodeType == CellType_Memory) {

                AlienGui::BeginIndent();

                // Mode selection
                auto& memory = std::get<MemoryGenomeDesc>(node._cellType);
                auto mode = memory.getMode();
                if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::MemoryModeStrings).textWidth(rightColumnWidth), mode)) {
                    memory._mode = createMemoryModeGenomeDesc(mode);
                    if (mode == MemoryMode_SignalRecorder || mode == MemoryMode_SignalStorage) {
                        memory._signalEntries.resize(8, SignalEntryGenomeDesc());
                    }
                }

                // Mode-specific parameters
                if (mode == MemoryMode_SignalDelay) {
                    AlienGui::BeginIndent();
                    auto& signalDelay = std::get<SignalDelayGenomeDesc>(memory._mode);
                    AlienGui::InputInt(AlienGui::InputIntParameters().name("Delay").textWidth(rightColumnWidth), signalDelay._delay);
                    AlienGui::EndIndent();
                } else if (mode == MemoryMode_SignalRecorder) {
                    AlienGui::BeginIndent();
                    auto& signalRecorder = std::get<SignalRecorderGenomeDesc>(memory._mode);
                    AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Read only").textWidth(rightColumnWidth), signalRecorder._readOnly);
                    AlienGui::SignalMemoryEditor(AlienGui::SignalMemoryEditorParameters().textWidth(rightColumnWidth), memory._signalEntries);
                    AlienGui::EndIndent();
                } else if (mode == MemoryMode_SignalStorage) {
                    AlienGui::BeginIndent();
                    auto& signalStorage = std::get<SignalStorageGenomeDesc>(memory._mode);
                    AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Read only").textWidth(rightColumnWidth), signalStorage._readOnly);
                    AlienGui::SignalMemoryEditor(AlienGui::SignalMemoryEditorParameters().textWidth(rightColumnWidth), memory._signalEntries);
                    AlienGui::EndIndent();
                } else if (mode == MemoryMode_SignalIntegrator) {
                    AlienGui::BeginIndent();
                    auto& signalIntegrator = std::get<SignalIntegratorGenomeDesc>(memory._mode);
                    AlienGui::SliderFloat(
                        AlienGui::SliderFloatParameters().name("New signal weight").max(1.0f).format("%.2f").textWidth(rightColumnWidth),
                        &signalIntegrator._newSignalWeight);
                    AlienGui::EndIndent();
                }

                bool bit[MAX_CHANNELS];
                for (int i = 0; i < MAX_CHANNELS; ++i) {
                    bit[i] = (memory._channelBitMask & (1 << i)) != 0;
                }
                AlienGui::MultiCheckboxes(
                    AlienGui::MultiCheckboxesParameters().name("Channel mask bit 0-3").textWidth(rightColumnWidth), bit[0], bit[1], bit[2], bit[3]);
                AlienGui::MultiCheckboxes(
                    AlienGui::MultiCheckboxesParameters().name("Channel mask bit 4-7").textWidth(rightColumnWidth), bit[4], bit[5], bit[6], bit[7]);
                AlienGui::MultiCheckboxes(
                    AlienGui::MultiCheckboxesParameters().name("Channel mask bit 8-11").textWidth(rightColumnWidth), bit[8], bit[9], bit[10], bit[11]);
                AlienGui::MultiCheckboxes(
                    AlienGui::MultiCheckboxesParameters().name("Channel mask bit 12-15").textWidth(rightColumnWidth), bit[12], bit[13], bit[14], bit[15]);
                memory._channelBitMask = 0;
                for (int i = 0; i < MAX_CHANNELS; ++i) {
                    if (bit[i]) {
                        memory._channelBitMask |= 1 << i;
                    }
                }

                AlienGui::EndIndent();
            } else if (nodeType == CellType_Communicator) {

                AlienGui::BeginIndent();

                // Mode selection
                auto& communicator = std::get<CommunicatorGenomeDesc>(node._cellType);
                auto mode = communicator.getMode();
                if (AlienGui::Combo(AlienGui::ComboParameters().name("Mode").values(Const::CommunicatorModeStrings).textWidth(rightColumnWidth), mode)) {
                    communicator._mode = createCommunicatorModeGenomeDesc(mode);
                }

                // Mode-specific parameters
                if (mode == CommunicatorMode_Sender) {
                    AlienGui::BeginIndent();
                    auto& sender = std::get<SenderGenomeDesc>(communicator._mode);
                    AlienGui::InputFloat(AlienGui::InputFloatParameters().name("Range").format("%.1f").textWidth(rightColumnWidth), sender._range);
                    AlienGui::InputInt(AlienGui::InputIntParameters().name("Max times sent").textWidth(rightColumnWidth), sender._maxTimesSent);
                    AlienGui::EndIndent();
                } else if (mode == CommunicatorMode_Receiver) {
                    AlienGui::BeginIndent();
                    auto& receiver = std::get<ReceiverGenomeDesc>(communicator._mode);
                    AlienGui::ComboOptionalColor(
                        AlienGui::ComboColorParameters().name("Restrict to color").textWidth(rightColumnWidth), receiver._restrictToColor);
                    AlienGui::Combo(
                        AlienGui::ComboParameters().name("Restrict to lineage").values({"No", "Same lineage", "Other lineage"}).textWidth(rightColumnWidth),
                        receiver._restrictToLineage);

                    AlienGui::EndIndent();
                }

                AlienGui::EndIndent();
            }

            // Optional constructor field (available for all node types)
            table.next();

            AlienGui::Group(AlienGui::GroupParameters().text("Construction properties"));

            bool hasConstructor = node._constructor.has_value();
            if (AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Has constructor").textWidth(rightColumnWidth), hasConstructor)) {
                if (hasConstructor) {
                    node._constructor = ConstructorGenomeDesc();
                } else {
                    node._constructor = std::nullopt;
                }
            }
            if (hasConstructor) {
                ImGui::PushID("Constructor");
                AlienGui::BeginIndent();
                auto& constructor = node._constructor.value();

                // Gene index
                std::vector<std::string> genes;
                for (auto const& [index, gene] : _editData->genome._genes | boost::adaptors::indexed(0)) {
                    auto text = "No. " + std::to_string(index + 1);
                    if (index == 0) {
                        text += " (root)";
                    }
                    genes.emplace_back(text);
                }
                AlienGui::Combo(AlienGui::ComboParameters().name("Gene").values(genes).textWidth(rightColumnWidth), constructor._geneIndex);

                // Auto activation interval
                AlienGui::InputOptionalInt(
                    AlienGui::InputIntParameters().name("Auto trigger interval").textWidth(rightColumnWidth), constructor._autoTriggerInterval);

                // Construction activation time
                AlienGui::InputInt(
                    AlienGui::InputIntParameters().name("Offspring trigger time").textWidth(rightColumnWidth), constructor._constructionActivationTime);

                // Construction angle
                AlienGui::InputFloat(
                    AlienGui::InputFloatParameters().name("Construction angle").textWidth(rightColumnWidth).format("%.1f"), constructor._constructionAngle);

                // Provide energy at construction
                auto provideEnergy = constructor._provideEnergy == ProvideEnergy_CellAndGene;
                AlienGui::Checkbox(AlienGui::CheckboxParameters().name("Provide energy").textWidth(rightColumnWidth), provideEnergy);
                constructor._provideEnergy = provideEnergy ? ProvideEnergy_CellAndGene : ProvideEnergy_CellOnly;

                AlienGui::EndIndent();
                ImGui::PopID();
            }

            table.next();
            table.end();
        }
    }
    ImGui::EndChild();
}

void _NodeEditorWidget::processNoSelection()
{
    AlienGui::Group(AlienGui::GroupParameters().text("Selected node"));
    if (ImGui::BeginChild("overlay", ImVec2(0, 0), 0)) {
        auto startPos = ImGui::GetCursorScreenPos();
        auto size = ImGui::GetContentRegionAvail();
        AlienGui::DisabledField();
        auto text = "No node is selected";
        auto textSize = ImGui::CalcTextSize(text);
        ImVec2 textPos(startPos.x + size.x / 2 - textSize.x / 2, startPos.y + size.y / 2 - textSize.y / 2);
        ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), text);
    }
    ImGui::EndChild();
}

void _NodeEditorWidget::processNeuralNetEditor()
{
    AlienGui::MoveTickUp();
    AlienGui::MoveTickUp();
    AlienGui::Group(AlienGui::GroupParameters().text("Neural network"));

    auto& node = _editData->getSelectedNodeRef();
    _neuralNetWidget->process(
        node._neuralNetwork._weights, node._neuralNetwork._biases, node._neuralNetwork._activationFunctions, node._neuralNetwork._connectionWeights);
}
