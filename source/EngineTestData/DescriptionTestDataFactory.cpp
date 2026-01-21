#include "DescriptionTestDataFactory.h"

#include <algorithm>

#include <boost/range/combine.hpp>

#include "TestHelper.h"

std::vector<DescriptionTestDataFactory::ObjectParameter> DescriptionTestDataFactory::getAllObjectParameters() const
{
    return {
        ObjectParameter{ObjectType_Structure},
        ObjectParameter{ObjectType_FreeCell},
        ObjectParameter{ObjectType_Cell, CellType_Base},
        ObjectParameter{ObjectType_Cell, CellType_Depot},
        ObjectParameter{ObjectType_Cell, CellType_Sensor, SensorModeWrapper{SensorMode_Telemetry}},
        ObjectParameter{ObjectType_Cell, CellType_Sensor, SensorModeWrapper{SensorMode_DetectEnergy}},
        ObjectParameter{ObjectType_Cell, CellType_Sensor, SensorModeWrapper{SensorMode_DetectStructure}},
        ObjectParameter{ObjectType_Cell, CellType_Sensor, SensorModeWrapper{SensorMode_DetectFreeCell}},
        ObjectParameter{ObjectType_Cell, CellType_Sensor, SensorModeWrapper{SensorMode_DetectCreature}},
        ObjectParameter{ObjectType_Cell, CellType_Generator},
        ObjectParameter{ObjectType_Cell, CellType_Attacker},
        ObjectParameter{ObjectType_Cell, CellType_Injector},
        ObjectParameter{ObjectType_Cell, CellType_Muscle, MuscleModeWrapper{MuscleMode_AutoBending}},
        ObjectParameter{ObjectType_Cell, CellType_Muscle, MuscleModeWrapper{MuscleMode_ManualBending}},
        ObjectParameter{ObjectType_Cell, CellType_Muscle, MuscleModeWrapper{MuscleMode_AngleBending}},
        ObjectParameter{ObjectType_Cell, CellType_Muscle, MuscleModeWrapper{MuscleMode_AutoCrawling}},
        ObjectParameter{ObjectType_Cell, CellType_Muscle, MuscleModeWrapper{MuscleMode_ManualCrawling}},
        ObjectParameter{ObjectType_Cell, CellType_Muscle, MuscleModeWrapper{MuscleMode_DirectMovement}},
        ObjectParameter{ObjectType_Cell, CellType_Defender},
        ObjectParameter{ObjectType_Cell, CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Structure}},
        ObjectParameter{ObjectType_Cell, CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_FreeCell}},
        ObjectParameter{ObjectType_Cell, CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Creature}},
        ObjectParameter{ObjectType_Cell, CellType_Detonator},
        ObjectParameter{ObjectType_Cell, CellType_Digestor},
        ObjectParameter{ObjectType_Cell, CellType_Memory, MemoryModeWrapper{MemoryMode_SignalDelay}},
        ObjectParameter{ObjectType_Cell, CellType_Memory, MemoryModeWrapper{MemoryMode_SignalRecorder}},
        ObjectParameter{ObjectType_Cell, CellType_Memory, MemoryModeWrapper{MemoryMode_SignalStorage}},
        ObjectParameter{ObjectType_Cell, CellType_Memory, MemoryModeWrapper{MemoryMode_SignalIntegrator}},
        ObjectParameter{ObjectType_Cell, CellType_Communicator, CommunicatorModeWrapper{CommunicatorMode_Sender}},
        ObjectParameter{ObjectType_Cell, CellType_Communicator, CommunicatorModeWrapper{CommunicatorMode_Receiver}},
    };
}

ObjectDesc DescriptionTestDataFactory::createNonDefaultObjectDesc(ObjectParameter objectParameter) const
{
    switch (objectParameter.objectType) {
    case ObjectType_Structure:
        return ObjectDesc().pos({0.5f, 0.8f}).vel({-0.3f, 0.7f}).color(3).fixed(true).type(StructureDesc().energy(42.0f));
    case ObjectType_FreeCell:
        return ObjectDesc().pos({0.5f, 0.8f}).vel({-0.3f, 0.7f}).color(3).fixed(true).type(FreeCellDesc().energy(42.0f).age(7));
    case ObjectType_Cell: {
        auto cellTypeDesc = createNonDefaultCellTypeDesc(objectParameter);
        NeuralNetworkDesc nn;
        nn.weight(2, 1, 0.7f);
        nn._biases.at(1) = -0.4f;
        nn._activationFunctions.at(5) = 2 % ActivationFunction_Count;
        return ObjectDesc()
            .pos({0.5f, 0.8f})
            .vel({-0.3f, 0.7f})
            .color(3)
            .fixed(true)
            .type(CellDesc()
                      .neuralNetwork(nn)
                      .usableEnergy(150.0f)
                      .rawEnergy(12.5f)
                      .age(42)
                      .cellState(false)
                      .geneIndex(42)
                      .nodeIndex(13)
                      .frontAngleId(13)
                      .headCell(true)
                      .parentNodeIndex(14)
                      .signal(SignalDesc().channels({1, 0, 0.6f, 0, 0, 0, 0, 0}).numTimesSent(5))
                      .signalState(SignalState_Active)
                      .signalRestriction(SignalRestrictionDesc().mode(SignalRestrictionMode_Active).baseAngle(45.0f).openingAngle(120.0f))
                      .constructor(ConstructorDesc()
                                       .autoTriggerInterval(55)
                                       .geneIndex(1)
                                       .constructionActivationTime(95)
                                       .constructionAngle(25.0f)
                                       .provideEnergy(ProvideEnergy_CellAndGene)
                                       .currentNodeIndex(2)
                                       .currentConcatenation(1)
                                       .currentBranch(0))
                      .event(CellEvent_Attacking)
                      .eventCounter(3)
                      .eventPos({1.5f, 2.5f})
                      .cellType(cellTypeDesc));
    }
    default:
        CHECK(false);
    }
}

EnergyDesc DescriptionTestDataFactory::createNonDefaultEnergyDesc() const
{
    return EnergyDesc().id(1).pos({0.3f, 0.9f}).vel({-0.6f, 0.2f}).energy(75.0f).color(5);
}

std::vector<DescriptionTestDataFactory::NodeParameter> DescriptionTestDataFactory::getAllNodeParameters() const
{
    return {
        NodeParameter{CellType_Base},
        NodeParameter{CellType_Depot},
        NodeParameter{CellType_Sensor, SensorModeWrapper{SensorMode_Telemetry}},
        NodeParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectEnergy}},
        NodeParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectStructure}},
        NodeParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectFreeCell}},
        NodeParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectCreature}},
        NodeParameter{CellType_Generator},
        NodeParameter{CellType_Attacker},
        NodeParameter{CellType_Injector},
        NodeParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_AutoBending}},
        NodeParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_ManualBending}},
        NodeParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_AngleBending}},
        NodeParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_AutoCrawling}},
        NodeParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_ManualCrawling}},
        NodeParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_DirectMovement}},
        NodeParameter{CellType_Defender},
        NodeParameter{CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Structure}},
        NodeParameter{CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_FreeCell}},
        NodeParameter{CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Creature}},
        NodeParameter{CellType_Detonator},
        NodeParameter{CellType_Digestor},
        NodeParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalDelay}},
        NodeParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalRecorder}},
        NodeParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalStorage}},
        NodeParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalIntegrator}},
        NodeParameter{CellType_Communicator, CommunicatorModeWrapper{CommunicatorMode_Sender}},
        NodeParameter{CellType_Communicator, CommunicatorModeWrapper{CommunicatorMode_Receiver}},
    };
}

NodeDesc DescriptionTestDataFactory::createNonDefaultNodeDesc(NodeParameter nodeParameter) const
{
    NeuralNetworkGenomeDesc nn;
    nn.weight(4, 3, 0.8f);
    nn._biases.at(3) = -0.5f;
    nn._activationFunctions.at(2) = 1;

    return NodeDesc()
        .neuralNetwork(nn)
        .cellType(createNonDefaultCellTypeGenomeDesc(nodeParameter))
        .constructor(ConstructorGenomeDesc().autoTriggerInterval(55).geneIndex(1).constructionActivationTime(95).constructionAngle(25.0f).provideEnergy(
            ProvideEnergy_FreeGeneration))
        .color(4)
        .numAdditionalConnections(3)
        .referenceAngle(90.0f)
        .signalRestriction(SignalRestrictionGenomeDesc().mode(SignalRestrictionMode_Active).baseAngle(60.0f).openingAngle(180.0f));
}

std::pair<CreatureDesc, GenomeDesc> DescriptionTestDataFactory::createNonDefaultCreatureDesc(NodeParameter nodeParameter) const
{
    auto genome = GenomeDesc()
                      .name("Test Genome")
                      .frontAngle(270.0f)
                      .genes({
                          GeneDesc()
                              .name("Test Gene")
                              .shape(ConstructorShape_Hexagon)
                              .numBranches(4)
                              .separation(true)
                              .numConcatenations(6)
                              .angleAlignment(ConstructorAngleAlignment_180)
                              .stiffness(0.75f)
                              .connectionDistance(0.8f)
                              .nodes({
                                  createNonDefaultNodeDesc(nodeParameter),
                              }),
                      });

    auto creature = CreatureDesc().ancestorId(1001).lineageId(502).generation(7).numObjects(25).frontAngleId(42).haveMutationsApplied(3).genomeId(genome._id);

    return {creature, genome};
}

bool DescriptionTestDataFactory::compare(Desc left, Desc right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(ObjectDesc left, ObjectDesc right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(EnergyDesc left, EnergyDesc right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(ObjectDesc const& object, NodeDesc const& node) const
{
    if (object._color != node._color) {
        return false;
    }

    // Object constructed via a genome must be a cell
    if (object.getObjectType() != ObjectType_Cell) {
        return false;
    }
    auto const& cell = object.getCellRef();

    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
        if (cell._neuralNetwork._weights[i] != node._neuralNetwork._weights[i]) {
            return false;
        }
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        if (cell._neuralNetwork._biases[i] != node._neuralNetwork._biases[i]) {
            return false;
        }
        if (cell._neuralNetwork._activationFunctions[i] != node._neuralNetwork._activationFunctions[i]) {
            return false;
        }
    }
    if (cell._signalRestriction._mode != node._signalRestriction._mode) {
        return false;
    }
    if (cell._signalRestriction._baseAngle != node._signalRestriction._baseAngle) {
        return false;
    }
    if (cell._signalRestriction._openingAngle != node._signalRestriction._openingAngle) {
        return false;
    }

    auto nodeType = node.getCellType();
    switch (cell.getCellType()) {
    case CellType_Base: {
        if (nodeType != CellType_Base) {
            return false;
        }
    } break;
    case CellType_Depot: {
        if (nodeType != CellType_Depot) {
            return false;
        }
        auto const& depot = std::get<DepotDesc>(cell._cellType);
        auto const& nodeDepot = std::get<DepotGenomeDesc>(node._cellType);
        if (depot._storageLimit != nodeDepot._storageLimit) {
            return false;
        }
    } break;
    case CellType_Sensor: {
        if (nodeType != CellType_Sensor) {
            return false;
        }
        auto const& sensor = std::get<SensorDesc>(cell._cellType);
        auto const& nodeSensor = std::get<SensorGenomeDesc>(node._cellType);
        if (sensor._autoTriggerInterval != nodeSensor._autoTriggerInterval) {
            return false;
        }
        if (sensor._minRange != nodeSensor._minRange) {
            return false;
        }
        if (sensor._maxRange != nodeSensor._maxRange) {
            return false;
        }
        // Compare modes
        if (sensor.getMode() != nodeSensor.getMode()) {
            return false;
        }
        // Compare mode-specific data
        switch (sensor.getMode()) {
        case SensorMode_DetectEnergy: {
            auto const& detectEnergy = std::get<DetectEnergyDesc>(sensor._mode);
            auto const& nodeDetectEnergy = std::get<DetectEnergyGenomeDesc>(nodeSensor._mode);
            if (detectEnergy._minDensity != nodeDetectEnergy._minDensity) {
                return false;
            }
        } break;
        case SensorMode_DetectStructure: {
            // No fields to compare
        } break;
        case SensorMode_DetectFreeCell: {
            auto const& detectFreeCell = std::get<DetectFreeCellDesc>(sensor._mode);
            auto const& nodeDetectFreeCell = std::get<DetectFreeCellGenomeDesc>(nodeSensor._mode);
            if (detectFreeCell._minDensity != nodeDetectFreeCell._minDensity) {
                return false;
            }
            if (detectFreeCell._restrictToColor != nodeDetectFreeCell._restrictToColor) {
                return false;
            }
        } break;
        case SensorMode_DetectCreature: {
            auto const& detectCreature = std::get<DetectCreatureDesc>(sensor._mode);
            auto const& nodeDetectCreature = std::get<DetectCreatureGenomeDesc>(nodeSensor._mode);
            if (detectCreature._minNumCells != nodeDetectCreature._minNumCells) {
                return false;
            }
            if (detectCreature._maxNumCells != nodeDetectCreature._maxNumCells) {
                return false;
            }
            if (detectCreature._restrictToColor != nodeDetectCreature._restrictToColor) {
                return false;
            }
            if (detectCreature._restrictToLineage != nodeDetectCreature._restrictToLineage) {
                return false;
            }
        } break;
        }
    } break;
    case CellType_Generator: {
        if (nodeType != CellType_Generator) {
            return false;
        }
        auto const& generator = std::get<GeneratorDesc>(cell._cellType);
        auto const& nodeGenerator = std::get<GeneratorGenomeDesc>(node._cellType);
        if (generator._autoTriggerInterval != nodeGenerator._autoTriggerInterval) {
            return false;
        }
        if (generator._pulseType != nodeGenerator._pulseType) {
            return false;
        }
        if (generator._alternationInterval != nodeGenerator._alternationInterval) {
            return false;
        }
    } break;
    case CellType_Attacker: {
        if (nodeType != CellType_Attacker) {
            return false;
        }
        auto const& attacker = std::get<AttackerDesc>(cell._cellType);
        auto const& nodeAttacker = std::get<AttackerGenomeDesc>(node._cellType);
        if (attacker.getMode() != nodeAttacker.getMode()) {
            return false;
        }
        switch (attacker.getMode()) {
        case AttackerMode_FreeCell: {
            auto const& freeCellMode = std::get<AttackFreeCellDesc>(attacker._mode);
            auto const& nodeFreeCellMode = std::get<AttackFreeCellGenomeDesc>(nodeAttacker._mode);
            if (freeCellMode._restrictToColor != nodeFreeCellMode._restrictToColor) {
                return false;
            }
        } break;
        case AttackerMode_Creature: {
            auto const& creatureMode = std::get<AttackCreatureDesc>(attacker._mode);
            auto const& nodeCreatureMode = std::get<AttackCreatureGenomeDesc>(nodeAttacker._mode);
            if (creatureMode._minNumCells != nodeCreatureMode._minNumCells) {
                return false;
            }
            if (creatureMode._maxNumCells != nodeCreatureMode._maxNumCells) {
                return false;
            }
            if (creatureMode._restrictToColor != nodeCreatureMode._restrictToColor) {
                return false;
            }
            if (creatureMode._restrictToLineage != nodeCreatureMode._restrictToLineage) {
                return false;
            }
        } break;
        }
    } break;
    case CellType_Injector: {
        if (nodeType != CellType_Injector) {
            return false;
        }
        auto const& injector = std::get<InjectorDesc>(cell._cellType);
        auto const& nodeInjector = std::get<InjectorGenomeDesc>(node._cellType);
        if (injector._geneIndex != nodeInjector._geneIndex) {
            return false;
        }
    } break;
    case CellType_Muscle: {
        if (nodeType != CellType_Muscle) {
            return false;
        }
        auto const& muscle = std::get<MuscleDesc>(cell._cellType);
        auto const& nodeMuscle = std::get<MuscleGenomeDesc>(node._cellType);
        if (muscle.getMode() != nodeMuscle.getMode()) {
            return false;
        }
        switch (muscle.getMode()) {
        case MuscleMode_AutoBending: {
            auto const& autoBending = std::get<AutoBendingDesc>(muscle._mode);
            auto const& nodeAutoBending = std::get<AutoBendingGenomeDesc>(nodeMuscle._mode);
            if (autoBending._maxAngleDeviation != nodeAutoBending._maxAngleDeviation) {
                return false;
            }
            if (autoBending._forwardBackwardRatio != nodeAutoBending._forwardBackwardRatio) {
                return false;
            }
        } break;
        case MuscleMode_ManualBending: {
            auto const& manualBending = std::get<ManualBendingDesc>(muscle._mode);
            auto const& nodeManualBending = std::get<ManualBendingGenomeDesc>(nodeMuscle._mode);
            if (manualBending._maxAngleDeviation != nodeManualBending._maxAngleDeviation) {
                return false;
            }
            if (manualBending._forwardBackwardRatio != nodeManualBending._forwardBackwardRatio) {
                return false;
            }
        } break;
        case MuscleMode_AngleBending: {
            auto const& angleBending = std::get<AngleBendingDesc>(muscle._mode);
            auto const& nodeAngleBending = std::get<AngleBendingGenomeDesc>(nodeMuscle._mode);
            if (angleBending._maxAngleDeviation != nodeAngleBending._maxAngleDeviation) {
                return false;
            }
            if (angleBending._attractionRepulsionRatio != nodeAngleBending._attractionRepulsionRatio) {
                return false;
            }
        } break;
        case MuscleMode_AutoCrawling: {
            auto const& autoCrawling = std::get<AutoCrawlingDesc>(muscle._mode);
            auto const& nodeAutoCrawling = std::get<AutoCrawlingGenomeDesc>(nodeMuscle._mode);
            if (autoCrawling._maxDistanceDeviation != nodeAutoCrawling._maxDistanceDeviation) {
                return false;
            }
            if (autoCrawling._forwardBackwardRatio != nodeAutoCrawling._forwardBackwardRatio) {
                return false;
            }
        } break;
        case MuscleMode_ManualCrawling: {
            auto const& manualCrawling = std::get<ManualCrawlingDesc>(muscle._mode);
            auto const& nodeManualCrawling = std::get<ManualCrawlingGenomeDesc>(nodeMuscle._mode);
            if (manualCrawling._maxDistanceDeviation != nodeManualCrawling._maxDistanceDeviation) {
                return false;
            }
            if (manualCrawling._forwardBackwardRatio != nodeManualCrawling._forwardBackwardRatio) {
                return false;
            }
        } break;
        case MuscleMode_DirectMovement: {
        } break;
        default:
            return false;
        }
    } break;
    case CellType_Defender: {
        if (nodeType != CellType_Defender) {
            return false;
        }
        auto const& defender = std::get<DefenderDesc>(cell._cellType);
        auto const& nodeDefender = std::get<DefenderGenomeDesc>(node._cellType);
        if (defender._mode != nodeDefender._mode) {
            return false;
        }
    } break;
    case CellType_Reconnector: {
        if (nodeType != CellType_Reconnector) {
            return false;
        }
        auto const& reconnector = std::get<ReconnectorDesc>(cell._cellType);
        auto const& nodeReconnector = std::get<ReconnectorGenomeDesc>(node._cellType);
        if (reconnector.getMode() != nodeReconnector.getMode()) {
            return false;
        }
        switch (reconnector.getMode()) {
        case ReconnectorMode_Structure: {
            // No fields to compare
        } break;
        case ReconnectorMode_FreeCell: {
            auto const& freeCellMode = std::get<ReconnectFreeCellDesc>(reconnector._mode);
            auto const& nodeFreeCellMode = std::get<ReconnectFreeCellGenomeDesc>(nodeReconnector._mode);
            if (freeCellMode._restrictToColor != nodeFreeCellMode._restrictToColor) {
                return false;
            }
        } break;
        case ReconnectorMode_Creature: {
            auto const& creatureMode = std::get<ReconnectCreatureDesc>(reconnector._mode);
            auto const& nodeCreatureMode = std::get<ReconnectCreatureGenomeDesc>(nodeReconnector._mode);
            if (creatureMode._minNumCells != nodeCreatureMode._minNumCells) {
                return false;
            }
            if (creatureMode._maxNumCells != nodeCreatureMode._maxNumCells) {
                return false;
            }
            if (creatureMode._restrictToColor != nodeCreatureMode._restrictToColor) {
                return false;
            }
            if (creatureMode._restrictToLineage != nodeCreatureMode._restrictToLineage) {
                return false;
            }
        } break;
        }
    } break;
    case CellType_Detonator: {
        if (nodeType != CellType_Detonator) {
            return false;
        }
        auto const& detonator = std::get<DetonatorDesc>(cell._cellType);
        auto const& nodeDetonator = std::get<DetonatorGenomeDesc>(node._cellType);
        if (detonator._countdown != nodeDetonator._countdown) {
            return false;
        }
    } break;
    case CellType_Digestor: {
        if (nodeType != CellType_Digestor) {
            return false;
        }
        auto const& digestor = std::get<DigestorDesc>(cell._cellType);
        auto const& nodeDigestor = std::get<DigestorGenomeDesc>(node._cellType);
        if (digestor._rawEnergyConductivity != nodeDigestor._rawEnergyConductivity) {
            return false;
        }
    } break;
    case CellType_Memory: {
        if (nodeType != CellType_Memory) {
            return false;
        }
        auto const& memory = std::get<MemoryDesc>(cell._cellType);
        auto const& nodeMemory = std::get<MemoryGenomeDesc>(node._cellType);
        if (memory.getMode() != nodeMemory.getMode()) {
            return false;
        }
        if (memory._channelBitMask != nodeMemory._channelBitMask) {
            return false;
        }
        switch (memory.getMode()) {
        case MemoryMode_SignalDelay: {
            auto const& signalDelay = std::get<SignalDelayDesc>(memory._mode);
            auto const& nodeSignalDelay = std::get<SignalDelayGenomeDesc>(nodeMemory._mode);
            if (signalDelay._delay != nodeSignalDelay._delay) {
                return false;
            }
        } break;
        case MemoryMode_SignalRecorder: {
            auto const& signalRecorder = std::get<SignalRecorderDesc>(memory._mode);
            auto const& nodeSignalRecorder = std::get<SignalRecorderGenomeDesc>(nodeMemory._mode);
            if (signalRecorder._readOnly != nodeSignalRecorder._readOnly) {
                return false;
            }
            if (signalRecorder._numWrittenSignalEntries != nodeSignalRecorder._numWrittenSignalEntries) {
                return false;
            }
        } break;
        case MemoryMode_SignalStorage: {
            auto const& signalStorage = std::get<SignalStorageDesc>(memory._mode);
            auto const& nodeSignalStorage = std::get<SignalStorageGenomeDesc>(nodeMemory._mode);
            if (signalStorage._readOnly != nodeSignalStorage._readOnly) {
                return false;
            }
        } break;
        case MemoryMode_SignalIntegrator: {
            auto const& signalIntegrator = std::get<SignalIntegratorDesc>(memory._mode);
            auto const& nodeSignalIntegrator = std::get<SignalIntegratorGenomeDesc>(nodeMemory._mode);
            if (signalIntegrator._newSignalWeight != nodeSignalIntegrator._newSignalWeight) {
                return false;
            }
        } break;
        }
        if (memory._signalEntries.size() != nodeMemory._signalEntries.size()) {
            return false;
        }
        for (auto const& [entry, nodeEntry] : boost::combine(memory._signalEntries, nodeMemory._signalEntries)) {
            for (auto const& [channel, nodeChannel] : boost::combine(entry._channels, nodeEntry._channels)) {
                if (channel != nodeChannel) {
                    return false;
                }
            }
        }
    } break;
    case CellType_Communicator: {
        if (nodeType != CellType_Communicator) {
            return false;
        }
        auto const& communicator = std::get<CommunicatorDesc>(cell._cellType);
        auto const& nodeCommunicator = std::get<CommunicatorGenomeDesc>(node._cellType);
        if (communicator.getMode() != nodeCommunicator.getMode()) {
            return false;
        }
        switch (communicator.getMode()) {
        case CommunicatorMode_Sender: {
            auto const& sender = std::get<SenderDesc>(communicator._mode);
            auto const& nodeSender = std::get<SenderGenomeDesc>(nodeCommunicator._mode);
            if (sender._range != nodeSender._range) {
                return false;
            }
            if (sender._maxTimesSent != nodeSender._maxTimesSent) {
                return false;
            }
        } break;
        case CommunicatorMode_Receiver: {
            auto const& receiver = std::get<ReceiverDesc>(communicator._mode);
            auto const& nodeReceiver = std::get<ReceiverGenomeDesc>(nodeCommunicator._mode);
            if (receiver._restrictToColor != nodeReceiver._restrictToColor) {
                return false;
            }
            if (receiver._restrictToLineage != nodeReceiver._restrictToLineage) {
                return false;
            }
        } break;
        }
    } break;
    default:
        return false;
    }

    // Compare optional constructor fields
    if (cell._constructor.has_value() != node._constructor.has_value()) {
        return false;
    }
    if (cell._constructor.has_value()) {
        auto const& constructor = cell._constructor.value();
        auto const& nodeConstructor = node._constructor.value();
        if (constructor._autoTriggerInterval != nodeConstructor._autoTriggerInterval) {
            return false;
        }
        if (constructor._geneIndex != nodeConstructor._geneIndex) {
            return false;
        }
        if (constructor._constructionActivationTime != nodeConstructor._constructionActivationTime) {
            return false;
        }
        if (constructor._constructionAngle != nodeConstructor._constructionAngle) {
            return false;
        }
        if (constructor._provideEnergy != nodeConstructor._provideEnergy) {
            return false;
        }
    }

    return true;
}

CellTypeDesc DescriptionTestDataFactory::createNonDefaultCellTypeDesc(ObjectParameter objectParameter) const
{
    auto const& type = objectParameter.cellType;
    auto muscleMode =
        std::holds_alternative<MuscleModeWrapper>(objectParameter.mode) ? std::get<MuscleModeWrapper>(objectParameter.mode).value : MuscleMode_AutoBending;
    auto sensorMode =
        std::holds_alternative<SensorModeWrapper>(objectParameter.mode) ? std::get<SensorModeWrapper>(objectParameter.mode).value : SensorMode_DetectEnergy;
    auto reconnectorMode = std::holds_alternative<ReconnectorModeWrapper>(objectParameter.mode) ? std::get<ReconnectorModeWrapper>(objectParameter.mode).value
                                                                                                : ReconnectorMode_Structure;
    auto memoryMode =
        std::holds_alternative<MemoryModeWrapper>(objectParameter.mode) ? std::get<MemoryModeWrapper>(objectParameter.mode).value : MemoryMode_SignalDelay;

    switch (type) {
    case CellType_Base:
        return BaseDesc();
    case CellType_Depot:
        return DepotDesc().storageLimit(300.0f).storedUsableEnergy(50.0f);
    case CellType_Sensor: {
        SensorModeDesc sensorModeDesc;
        switch (sensorMode) {
        case SensorMode_Telemetry:
            sensorModeDesc = TelemetryDesc();
            break;
        case SensorMode_DetectEnergy:
            sensorModeDesc = DetectEnergyDesc().minDensity(0.3f);
            break;
        case SensorMode_DetectStructure:
            sensorModeDesc = DetectStructureDesc();
            break;
        case SensorMode_DetectFreeCell:
            sensorModeDesc = DetectFreeCellDesc().minDensity(0.25f).restrictToColor(2);
            break;
        case SensorMode_DetectCreature:
            sensorModeDesc = DetectCreatureDesc().minNumCells(5).maxNumCells(20).restrictToColor(3).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            sensorModeDesc = SensorModeDesc();
            break;
        }
        return SensorDesc()
            .autoTriggerInterval(80)
            .mode(sensorModeDesc)
            .minRange(10)
            .maxRange(50)
            .lastMatch(SensorLastMatchDesc().creatureId(42).pos({10.5f, 20.3f}));
    }
    case CellType_Generator: {
        return GeneratorDesc().autoTriggerInterval(60).alternationInterval(3).numPulses(5);
    }
    case CellType_Attacker:
        return AttackerDesc().mode(AttackCreatureDesc().minNumCells(4).maxNumCells(15).restrictToColor(1).restrictToLineage(LineageRestriction_OtherLineage));
    case CellType_Injector:
        return InjectorDesc().geneIndex(3);
    case CellType_Muscle: {
        MuscleModeDesc muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending: {
            muscleModeDesc = AutoBendingDesc()
                                 .maxAngleDeviation(0.6f)
                                 .forwardBackwardRatio(0.4f)
                                 .initialAngle(135.0f)
                                 .forward(false)
                                 .activation(0.7f)
                                 .activationCountdown(8)
                                 .impulseAlreadyApplied(true);
        } break;
        case MuscleMode_ManualBending:
            muscleModeDesc =
                ManualBendingDesc().maxAngleDeviation(0.5f).forwardBackwardRatio(0.3f).initialAngle(225.0f).lastAngleDelta(0.8f).impulseAlreadyApplied(true);
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingDesc().maxAngleDeviation(0.7f).attractionRepulsionRatio(0.6f).initialAngle(315.0f);
            break;
        case MuscleMode_AutoCrawling: {
            AutoCrawlingDesc defaultCrawling;
            muscleModeDesc = AutoCrawlingDesc()
                                 .maxDistanceDeviation(0.9f)
                                 .forwardBackwardRatio(0.35f)
                                 .initialDistance(0.6f)
                                 .lastActualDistance(0.8f)
                                 .forward(false)
                                 .activation(0.4f)
                                 .activationCountdown(12)
                                 .impulseAlreadyApplied(true);
        } break;
        case MuscleMode_ManualCrawling:
            muscleModeDesc = ManualCrawlingDesc()
                                 .maxDistanceDeviation(0.75f)
                                 .forwardBackwardRatio(0.45f)
                                 .initialDistance(0.4f)
                                 .lastActualDistance(0.9f)
                                 .lastDistanceDelta(0.65f)
                                 .impulseAlreadyApplied(true);
            break;
        case MuscleMode_DirectMovement:
            muscleModeDesc = DirectMovementDesc();
            break;
        default:
            muscleModeDesc = MuscleModeDesc();
        }
        return MuscleDesc().mode(muscleModeDesc);
    }
    case CellType_Defender:
        return DefenderDesc().mode(DefenderMode_DefendAgainstInjector);
    case CellType_Reconnector: {
        ReconnectorModeDesc reconnectorModeDesc;
        switch (reconnectorMode) {
        case ReconnectorMode_Structure:
            reconnectorModeDesc = ReconnectStructureDesc();
            break;
        case ReconnectorMode_FreeCell:
            reconnectorModeDesc = ReconnectFreeCellDesc().restrictToColor(2);
            break;
        case ReconnectorMode_Creature:
            reconnectorModeDesc = ReconnectCreatureDesc().minNumCells(5).maxNumCells(20).restrictToColor(3).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            reconnectorModeDesc = ReconnectorModeDesc();
        }
        return ReconnectorDesc().mode(reconnectorModeDesc);
    }
    case CellType_Detonator:
        return DetonatorDesc().countdown(23);
    case CellType_Digestor:
        return DigestorDesc().rawEnergyConductivity(0.7f);
    case CellType_Memory: {
        MemoryModeDesc memoryModeDesc;
        switch (memoryMode) {
        case MemoryMode_SignalDelay:
            memoryModeDesc = SignalDelayDesc().delay(15).numSignalEntriesInitialized(5).ringBufferIndex(3);
            break;
        case MemoryMode_SignalRecorder:
            memoryModeDesc = SignalRecorderDesc().readOnly(false).state(SignalRecorderState_Recording).numWrittenSignalEntries(3).numReadSignalEntries(1);
            break;
        case MemoryMode_SignalStorage:
            memoryModeDesc = SignalStorageDesc().readOnly(false);
            break;
        case MemoryMode_SignalIntegrator:
            memoryModeDesc = SignalIntegratorDesc().newSignalWeight(0.75f);
            break;
        default:
            memoryModeDesc = MemoryModeDesc();
        }
        auto memory = MemoryDesc().mode(memoryModeDesc).channelBitMask(0b01010101);
        for (int i = 0; i < 10; ++i) {
            SignalEntryDesc entry;
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                entry._channels[j] = static_cast<float>(i * MAX_CHANNELS + j) * 0.15f;
            }
            memory._signalEntries.emplace_back(entry);
        }
        return memory;
    }
    case CellType_Communicator: {
        auto communicatorMode = std::holds_alternative<CommunicatorModeWrapper>(objectParameter.mode)
            ? std::get<CommunicatorModeWrapper>(objectParameter.mode).value
            : CommunicatorMode_Sender;
        CommunicatorModeDesc communicatorModeDesc;
        switch (communicatorMode) {
        case CommunicatorMode_Sender:
            communicatorModeDesc = SenderDesc().range(150.0f).maxTimesSent(6);
            break;
        case CommunicatorMode_Receiver:
            communicatorModeDesc = ReceiverDesc().restrictToColor(2).restrictToLineage(LineageRestriction_OtherLineage);
            break;
        default:
            communicatorModeDesc = CommunicatorModeDesc();
        }
        return CommunicatorDesc().mode(communicatorModeDesc);
    }
    default:
        return CellTypeDesc();
    }
}

CellTypeGenomeDesc DescriptionTestDataFactory::createNonDefaultCellTypeGenomeDesc(NodeParameter objectParameter) const
{
    auto const& type = objectParameter.cellTypeGenome;
    auto muscleMode =
        std::holds_alternative<MuscleModeWrapper>(objectParameter.mode) ? std::get<MuscleModeWrapper>(objectParameter.mode).value : MuscleMode_AutoBending;
    auto sensorMode =
        std::holds_alternative<SensorModeWrapper>(objectParameter.mode) ? std::get<SensorModeWrapper>(objectParameter.mode).value : SensorMode_DetectEnergy;
    auto reconnectorMode = std::holds_alternative<ReconnectorModeWrapper>(objectParameter.mode) ? std::get<ReconnectorModeWrapper>(objectParameter.mode).value
                                                                                                : ReconnectorMode_Structure;
    auto memoryMode =
        std::holds_alternative<MemoryModeWrapper>(objectParameter.mode) ? std::get<MemoryModeWrapper>(objectParameter.mode).value : MemoryMode_SignalDelay;
    switch (type) {
    case CellType_Base:
        return BaseGenomeDesc();
    case CellType_Depot:
        return DepotGenomeDesc().storageLimit(350.0f).initialStoredUsableEnergy(100.0f);
    case CellType_Sensor: {
        SensorModeGenomeDesc sensorModeDesc;
        switch (sensorMode) {
        case SensorMode_Telemetry:
            sensorModeDesc = TelemetryGenomeDesc();
            break;
        case SensorMode_DetectEnergy:
            sensorModeDesc = DetectEnergyGenomeDesc().minDensity(0.25f);
            break;
        case SensorMode_DetectStructure:
            sensorModeDesc = DetectStructureGenomeDesc();
            break;
        case SensorMode_DetectFreeCell:
            sensorModeDesc = DetectFreeCellGenomeDesc().minDensity(0.20f).restrictToColor(6);
            break;
        case SensorMode_DetectCreature:
            sensorModeDesc = DetectCreatureGenomeDesc().minNumCells(3).maxNumCells(15).restrictToColor(4).restrictToLineage(LineageRestriction_OtherLineage);
            break;
        default:
            sensorModeDesc = SensorModeGenomeDesc();
            break;
        }
        return SensorGenomeDesc().autoTriggerInterval(70).mode(sensorModeDesc).minRange(5).maxRange(30);
    }
    case CellType_Generator:
        return GeneratorGenomeDesc().autoTriggerInterval(55).pulseType(GeneratorPulseType_Alternation).alternationInterval(4);
    case CellType_Attacker:
        return AttackerGenomeDesc().mode(
            AttackCreatureGenomeDesc().minNumCells(5).maxNumCells(18).restrictToColor(2).restrictToLineage(LineageRestriction_SameLineage));
    case CellType_Injector:
        return InjectorGenomeDesc().geneIndex(3);
    case CellType_Muscle: {
        MuscleModeGenomeDesc muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending:
            muscleModeDesc = AutoBendingGenomeDesc().maxAngleDeviation(0.55f).forwardBackwardRatio(0.25f);
            break;
        case MuscleMode_ManualBending:
            muscleModeDesc = ManualBendingGenomeDesc().maxAngleDeviation(0.45f).forwardBackwardRatio(0.35f);
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingGenomeDesc().maxAngleDeviation(0.65f).attractionRepulsionRatio(0.55f);
            break;
        case MuscleMode_AutoCrawling:
            muscleModeDesc = AutoCrawlingGenomeDesc().maxDistanceDeviation(0.85f).forwardBackwardRatio(0.15f);
            break;
        case MuscleMode_ManualCrawling:
            muscleModeDesc = ManualCrawlingGenomeDesc().maxDistanceDeviation(0.95f).forwardBackwardRatio(0.25f);
            break;
        case MuscleMode_DirectMovement:
            muscleModeDesc = DirectMovementGenomeDesc();
            break;
        default:
            muscleModeDesc = MuscleModeGenomeDesc();
        }
        return MuscleGenomeDesc().mode(muscleModeDesc);
    }
    case CellType_Defender:
        return DefenderGenomeDesc().mode(DefenderMode_DefendAgainstInjector);
    case CellType_Reconnector: {
        ReconnectorModeGenomeDesc reconnectorModeDesc;
        switch (reconnectorMode) {
        case ReconnectorMode_Structure:
            reconnectorModeDesc = ReconnectStructureGenomeDesc();
            break;
        case ReconnectorMode_FreeCell:
            reconnectorModeDesc = ReconnectFreeCellGenomeDesc().restrictToColor(6);
            break;
        case ReconnectorMode_Creature:
            reconnectorModeDesc =
                ReconnectCreatureGenomeDesc().minNumCells(5).maxNumCells(20).restrictToColor(4).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            reconnectorModeDesc = ReconnectorModeGenomeDesc();
        }
        return ReconnectorGenomeDesc().mode(reconnectorModeDesc);
    }
    case CellType_Detonator: {
        return DetonatorGenomeDesc().countdown(45);
    }
    case CellType_Digestor: {
        return DigestorGenomeDesc().rawEnergyConductivity(0.8f);
    }
    case CellType_Memory: {
        MemoryModeGenomeDesc memoryModeDesc;
        switch (memoryMode) {
        case MemoryMode_SignalDelay:
            memoryModeDesc = SignalDelayGenomeDesc().delay(20);
            break;
        case MemoryMode_SignalRecorder:
            memoryModeDesc = SignalRecorderGenomeDesc().readOnly(false).numWrittenSignalEntries(3);
            break;
        case MemoryMode_SignalStorage:
            memoryModeDesc = SignalStorageGenomeDesc().readOnly(false);
            break;
        case MemoryMode_SignalIntegrator:
            memoryModeDesc = SignalIntegratorGenomeDesc().newSignalWeight(0.8f);
            break;
        default:
            memoryModeDesc = MemoryModeGenomeDesc();
        }
        auto memory = MemoryGenomeDesc().mode(memoryModeDesc).channelBitMask(0b10101010);
        for (int i = 0; i < 5; ++i) {
            SignalEntryGenomeDesc entry;
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                entry._channels[j] = static_cast<float>(i * MAX_CHANNELS + j) * 0.15f;
            }
            memory._signalEntries.emplace_back(entry);
        }
        return memory;
    }
    case CellType_Communicator: {
        auto communicatorMode = std::holds_alternative<CommunicatorModeWrapper>(objectParameter.mode)
            ? std::get<CommunicatorModeWrapper>(objectParameter.mode).value
            : CommunicatorMode_Sender;
        CommunicatorModeGenomeDesc communicatorModeDesc;
        switch (communicatorMode) {
        case CommunicatorMode_Sender:
            communicatorModeDesc = SenderGenomeDesc().range(200.0f).maxTimesSent(8);
            break;
        case CommunicatorMode_Receiver:
            communicatorModeDesc = ReceiverGenomeDesc().restrictToColor(5).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            communicatorModeDesc = CommunicatorModeGenomeDesc();
        }
        return CommunicatorGenomeDesc().mode(communicatorModeDesc);
    }
    default:
        return CellTypeGenomeDesc();
    }
}
