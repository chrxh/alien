#include "DescriptionTestDataFactory.h"

#include <algorithm>
#include <boost/range/combine.hpp>

#include "TestHelper.h"

std::vector<DescriptionTestDataFactory::CellParameter> DescriptionTestDataFactory::getAllCellParameters() const
{
    return {
        CellParameter{CellType_Structure},
        CellParameter{CellType_Free},
        CellParameter{CellType_Base},
        CellParameter{CellType_Depot},
        CellParameter{CellType_Constructor},
        CellParameter{CellType_Sensor, SensorModeWrapper{SensorMode_Telemetry}},
        CellParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectEnergy}},
        CellParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectStructure}},
        CellParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectFreeCell}},
        CellParameter{CellType_Sensor, SensorModeWrapper{SensorMode_DetectCreature}},
        CellParameter{CellType_Generator},
        CellParameter{CellType_Attacker},
        CellParameter{CellType_Injector},
        CellParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_AutoBending}},
        CellParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_ManualBending}},
        CellParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_AngleBending}},
        CellParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_AutoCrawling}},
        CellParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_ManualCrawling}},
        CellParameter{CellType_Muscle, MuscleModeWrapper{MuscleMode_DirectMovement}},
        CellParameter{CellType_Defender},
        CellParameter{CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Structure}},
        CellParameter{CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_FreeCell}},
        CellParameter{CellType_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Creature}},
        CellParameter{CellType_Detonator},
        CellParameter{CellType_Digestor},
        CellParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalDelay}},
        CellParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalRecorder}},
        CellParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalStorage}},
        CellParameter{CellType_Memory, MemoryModeWrapper{MemoryMode_SignalIntegrator}},
        CellParameter{CellType_Communicator, CommunicatorModeWrapper{CommunicatorMode_Sender}},
        CellParameter{CellType_Communicator, CommunicatorModeWrapper{CommunicatorMode_Receiver}},
    };
}

CellDescription DescriptionTestDataFactory::createNonDefaultCellDescription(CellParameter cellParameter) const
{
    CellDescription defaultCell;

    auto cellTypeDesc = createNonDefaultCellTypeDescription(cellParameter);
    auto result = CellDescription()
                      .pos({0.5f, 0.8f})
                      .vel({-0.3f, 0.7f})
                      .usableEnergy(150.0f)
                      .rawEnergy(12.5f)
                      .age(42)
                      .color(3)
                      .fixed(true)
                      .cellState(false)
                      .geneIndex(42)
                      .nodeIndex(13)
                      .frontAngleId(13)
                      .headCell(true)
                      .parentNodeIndex(14)
                      .signal(SignalDescription().channels({1, 0, 0.6f, 0, 0, 0, 0, 0}).numTimesSent(5))
                      .signalState(SignalState_Active)
                      .signalRestriction(SignalRestrictionDescription().mode(SignalRestrictionMode_Active).baseAngle(45.0f).openingAngle(120.0f))
                      .cellType(cellTypeDesc);

    if (cellParameter.cellType != CellType_Structure && cellParameter.cellType != CellType_Free) {
        NeuralNetworkDescription defaultNn;
        NeuralNetworkDescription nn;
        nn.weight(2, 1, 0.7f);
        nn._biases.at(1) = -0.4f;
        nn._activationFunctions.at(5) = 2 % ActivationFunction_Count;
        result._neuralNetwork = nn;
    }
    return result;
}

ParticleDescription DescriptionTestDataFactory::createNonDefaultParticleDescription() const
{
    ParticleDescription defaultParticle;
    return ParticleDescription().id(1).pos({0.3f, 0.9f}).vel({-0.6f, 0.2f}).energy(75.0f).color(5);
}

std::vector<DescriptionTestDataFactory::NodeParameter> DescriptionTestDataFactory::getAllNodeParameters() const
{
    return {
        NodeParameter{CellTypeGenome_Base},
        NodeParameter{CellTypeGenome_Depot},
        NodeParameter{CellTypeGenome_Constructor},
        NodeParameter{CellTypeGenome_Sensor, SensorModeWrapper{SensorMode_Telemetry}},
        NodeParameter{CellTypeGenome_Sensor, SensorModeWrapper{SensorMode_DetectEnergy}},
        NodeParameter{CellTypeGenome_Sensor, SensorModeWrapper{SensorMode_DetectStructure}},
        NodeParameter{CellTypeGenome_Sensor, SensorModeWrapper{SensorMode_DetectFreeCell}},
        NodeParameter{CellTypeGenome_Sensor, SensorModeWrapper{SensorMode_DetectCreature}},
        NodeParameter{CellTypeGenome_Generator},
        NodeParameter{CellTypeGenome_Attacker},
        NodeParameter{CellTypeGenome_Injector},
        NodeParameter{CellTypeGenome_Muscle, MuscleModeWrapper{MuscleMode_AutoBending}},
        NodeParameter{CellTypeGenome_Muscle, MuscleModeWrapper{MuscleMode_ManualBending}},
        NodeParameter{CellTypeGenome_Muscle, MuscleModeWrapper{MuscleMode_AngleBending}},
        NodeParameter{CellTypeGenome_Muscle, MuscleModeWrapper{MuscleMode_AutoCrawling}},
        NodeParameter{CellTypeGenome_Muscle, MuscleModeWrapper{MuscleMode_ManualCrawling}},
        NodeParameter{CellTypeGenome_Muscle, MuscleModeWrapper{MuscleMode_DirectMovement}},
        NodeParameter{CellTypeGenome_Defender},
        NodeParameter{CellTypeGenome_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Structure}},
        NodeParameter{CellTypeGenome_Reconnector, ReconnectorModeWrapper{ReconnectorMode_FreeCell}},
        NodeParameter{CellTypeGenome_Reconnector, ReconnectorModeWrapper{ReconnectorMode_Creature}},
        NodeParameter{CellTypeGenome_Detonator},
        NodeParameter{CellTypeGenome_Digestor},
        NodeParameter{CellTypeGenome_Memory, MemoryModeWrapper{MemoryMode_SignalDelay}},
        NodeParameter{CellTypeGenome_Memory, MemoryModeWrapper{MemoryMode_SignalRecorder}},
        NodeParameter{CellTypeGenome_Memory, MemoryModeWrapper{MemoryMode_SignalStorage}},
        NodeParameter{CellTypeGenome_Memory, MemoryModeWrapper{MemoryMode_SignalIntegrator}},
        NodeParameter{CellTypeGenome_Communicator, CommunicatorModeWrapper{CommunicatorMode_Sender}},
        NodeParameter{CellTypeGenome_Communicator, CommunicatorModeWrapper{CommunicatorMode_Receiver}},
    };
}

NodeDescription DescriptionTestDataFactory::createNonDefaultNodeDescription(NodeParameter nodeParameter) const
{
    NodeDescription defaultNode;

    NeuralNetworkGenomeDescription nn;
    nn.weight(4, 3, 0.8f);
    nn._biases.at(3) = -0.5f;
    nn._activationFunctions.at(2) = 1;

    return NodeDescription()
        .neuralNetwork(nn)
        .cellType(createNonDefaultCellTypeGenomeDescription(nodeParameter))
        .color(4)
        .numAdditionalConnections(3)
        .referenceAngle(90.0f)
        .signalRestriction(SignalRestrictionGenomeDescription().mode(SignalRestrictionMode_Active).baseAngle(60.0f).openingAngle(180.0f));
}

std::pair<CreatureDescription, GenomeDescription> DescriptionTestDataFactory::createNonDefaultCreatureDescription(NodeParameter nodeParameter) const
{
    CreatureDescription defaultCreature;
    GeneDescription defaultGene;

    auto genome = GenomeDescription()
                      .name("Test Genome")
                      .frontAngle(270.0f)
                      .genes({
                          GeneDescription()
                              .name("Test Gene")
                              .shape(ConstructorShape_Hexagon)
                              .numBranches(4)
                              .separation(true)
                              .numConcatenations(6)
                              .angleAlignment(ConstructorAngleAlignment_180)
                              .stiffness(0.75f)
                              .connectionDistance(0.8f)
                              .nodes({
                                  createNonDefaultNodeDescription(nodeParameter),
                              }),
                      });

    auto creature = CreatureDescription().ancestorId(1001).lineageId(502).generation(7).numCells(25).frontAngleId(42).genomeId(genome._id);

    return {creature, genome};
}

bool DescriptionTestDataFactory::compare(Description left, Description right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(CellDescription left, CellDescription right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(ParticleDescription left, ParticleDescription right) const
{
    return TestHelper::compare(left, right);
}

bool DescriptionTestDataFactory::compare(CellDescription const& cell, NodeDescription const& node) const
{
    if (cell._color != node._color) {
        return false;
    }
    if (!cell._neuralNetwork.has_value()) {
        return false;
    }
    for (int i = 0; i < MAX_CHANNELS * MAX_CHANNELS; ++i) {
        if (cell._neuralNetwork->_weights[i] != node._neuralNetwork._weights[i]) {
            return false;
        }
    }
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        if (cell._neuralNetwork->_biases[i] != node._neuralNetwork._biases[i]) {
            return false;
        }
        if (cell._neuralNetwork->_activationFunctions[i] != node._neuralNetwork._activationFunctions[i]) {
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
        if (nodeType != CellTypeGenome_Base) {
            return false;
        }
    } break;
    case CellType_Depot: {
        if (nodeType != CellTypeGenome_Depot) {
            return false;
        }
        auto const& depot = std::get<DepotDescription>(cell._cellType);
        auto const& nodeDepot = std::get<DepotGenomeDescription>(node._cellType);
        if (depot._storageLimit != nodeDepot._storageLimit) {
            return false;
        }
    } break;
    case CellType_Constructor: {
        if (nodeType != CellTypeGenome_Constructor) {
            return false;
        }
        auto const& constructor = std::get<ConstructorDescription>(cell._cellType);
        auto const& nodeConstructor = std::get<ConstructorGenomeDescription>(node._cellType);
        if (constructor._autoTriggerInterval != nodeConstructor._autoTriggerInterval) {
            return false;
        }
        if (constructor._geneIndex != nodeConstructor._geneIndex) {
            return false;
        }
        if (constructor._constructionActivationTime != nodeConstructor._constructionActivationTime) {
            return false;
        }
    } break;
    case CellType_Sensor: {
        if (nodeType != CellTypeGenome_Sensor) {
            return false;
        }
        auto const& sensor = std::get<SensorDescription>(cell._cellType);
        auto const& nodeSensor = std::get<SensorGenomeDescription>(node._cellType);
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
            auto const& detectEnergy = std::get<DetectEnergyDescription>(sensor._mode);
            auto const& nodeDetectEnergy = std::get<DetectEnergyGenomeDescription>(nodeSensor._mode);
            if (detectEnergy._minDensity != nodeDetectEnergy._minDensity) {
                return false;
            }
        } break;
        case SensorMode_DetectStructure: {
            // No fields to compare
        } break;
        case SensorMode_DetectFreeCell: {
            auto const& detectFreeCell = std::get<DetectFreeCellDescription>(sensor._mode);
            auto const& nodeDetectFreeCell = std::get<DetectFreeCellGenomeDescription>(nodeSensor._mode);
            if (detectFreeCell._minDensity != nodeDetectFreeCell._minDensity) {
                return false;
            }
            if (detectFreeCell._restrictToColor != nodeDetectFreeCell._restrictToColor) {
                return false;
            }
        } break;
        case SensorMode_DetectCreature: {
            auto const& detectCreature = std::get<DetectCreatureDescription>(sensor._mode);
            auto const& nodeDetectCreature = std::get<DetectCreatureGenomeDescription>(nodeSensor._mode);
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
        if (nodeType != CellTypeGenome_Generator) {
            return false;
        }
        auto const& generator = std::get<GeneratorDescription>(cell._cellType);
        auto const& nodeGenerator = std::get<GeneratorGenomeDescription>(node._cellType);
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
        if (nodeType != CellTypeGenome_Attacker) {
            return false;
        }
        auto const& attacker = std::get<AttackerDescription>(cell._cellType);
        auto const& nodeAttacker = std::get<AttackerGenomeDescription>(node._cellType);
        if (attacker.getMode() != nodeAttacker.getMode()) {
            return false;
        }
        switch (attacker.getMode()) {
        case AttackerMode_FreeCell: {
            auto const& freeCellMode = std::get<AttackFreeCellDescription>(attacker._mode);
            auto const& nodeFreeCellMode = std::get<AttackFreeCellGenomeDescription>(nodeAttacker._mode);
            if (freeCellMode._restrictToColor != nodeFreeCellMode._restrictToColor) {
                return false;
            }
        } break;
        case AttackerMode_Creature: {
            auto const& creatureMode = std::get<AttackCreatureDescription>(attacker._mode);
            auto const& nodeCreatureMode = std::get<AttackCreatureGenomeDescription>(nodeAttacker._mode);
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
        if (nodeType != CellTypeGenome_Injector) {
            return false;
        }
        auto const& injector = std::get<InjectorDescription>(cell._cellType);
        auto const& nodeInjector = std::get<InjectorGenomeDescription>(node._cellType);
        if (injector._geneIndex != nodeInjector._geneIndex) {
            return false;
        }
    } break;
    case CellType_Muscle: {
        if (nodeType != CellTypeGenome_Muscle) {
            return false;
        }
        auto const& muscle = std::get<MuscleDescription>(cell._cellType);
        auto const& nodeMuscle = std::get<MuscleGenomeDescription>(node._cellType);
        if (muscle.getMode() != nodeMuscle.getMode()) {
            return false;
        }
        switch (muscle.getMode()) {
        case MuscleMode_AutoBending: {
            auto const& autoBending = std::get<AutoBendingDescription>(muscle._mode);
            auto const& nodeAutoBending = std::get<AutoBendingGenomeDescription>(nodeMuscle._mode);
            if (autoBending._maxAngleDeviation != nodeAutoBending._maxAngleDeviation) {
                return false;
            }
            if (autoBending._forwardBackwardRatio != nodeAutoBending._forwardBackwardRatio) {
                return false;
            }
        } break;
        case MuscleMode_ManualBending: {
            auto const& manualBending = std::get<ManualBendingDescription>(muscle._mode);
            auto const& nodeManualBending = std::get<ManualBendingGenomeDescription>(nodeMuscle._mode);
            if (manualBending._maxAngleDeviation != nodeManualBending._maxAngleDeviation) {
                return false;
            }
            if (manualBending._forwardBackwardRatio != nodeManualBending._forwardBackwardRatio) {
                return false;
            }
        } break;
        case MuscleMode_AngleBending: {
            auto const& angleBending = std::get<AngleBendingDescription>(muscle._mode);
            auto const& nodeAngleBending = std::get<AngleBendingGenomeDescription>(nodeMuscle._mode);
            if (angleBending._maxAngleDeviation != nodeAngleBending._maxAngleDeviation) {
                return false;
            }
            if (angleBending._attractionRepulsionRatio != nodeAngleBending._attractionRepulsionRatio) {
                return false;
            }
        } break;
        case MuscleMode_AutoCrawling: {
            auto const& autoCrawling = std::get<AutoCrawlingDescription>(muscle._mode);
            auto const& nodeAutoCrawling = std::get<AutoCrawlingGenomeDescription>(nodeMuscle._mode);
            if (autoCrawling._maxDistanceDeviation != nodeAutoCrawling._maxDistanceDeviation) {
                return false;
            }
            if (autoCrawling._forwardBackwardRatio != nodeAutoCrawling._forwardBackwardRatio) {
                return false;
            }
        } break;
        case MuscleMode_ManualCrawling: {
            auto const& manualCrawling = std::get<ManualCrawlingDescription>(muscle._mode);
            auto const& nodeManualCrawling = std::get<ManualCrawlingGenomeDescription>(nodeMuscle._mode);
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
        if (nodeType != CellTypeGenome_Defender) {
            return false;
        }
        auto const& defender = std::get<DefenderDescription>(cell._cellType);
        auto const& nodeDefender = std::get<DefenderGenomeDescription>(node._cellType);
        if (defender._mode != nodeDefender._mode) {
            return false;
        }
    } break;
    case CellType_Reconnector: {
        if (nodeType != CellTypeGenome_Reconnector) {
            return false;
        }
        auto const& reconnector = std::get<ReconnectorDescription>(cell._cellType);
        auto const& nodeReconnector = std::get<ReconnectorGenomeDescription>(node._cellType);
        if (reconnector.getMode() != nodeReconnector.getMode()) {
            return false;
        }
        switch (reconnector.getMode()) {
        case ReconnectorMode_Structure: {
            // No fields to compare
        } break;
        case ReconnectorMode_FreeCell: {
            auto const& freeCellMode = std::get<ReconnectFreeCellDescription>(reconnector._mode);
            auto const& nodeFreeCellMode = std::get<ReconnectFreeCellGenomeDescription>(nodeReconnector._mode);
            if (freeCellMode._restrictToColor != nodeFreeCellMode._restrictToColor) {
                return false;
            }
        } break;
        case ReconnectorMode_Creature: {
            auto const& creatureMode = std::get<ReconnectCreatureDescription>(reconnector._mode);
            auto const& nodeCreatureMode = std::get<ReconnectCreatureGenomeDescription>(nodeReconnector._mode);
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
        if (nodeType != CellTypeGenome_Detonator) {
            return false;
        }
        auto const& detonator = std::get<DetonatorDescription>(cell._cellType);
        auto const& nodeDetonator = std::get<DetonatorGenomeDescription>(node._cellType);
        if (detonator._countdown != nodeDetonator._countdown) {
            return false;
        }
    } break;
    case CellType_Digestor: {
        if (nodeType != CellTypeGenome_Digestor) {
            return false;
        }
        auto const& digestor = std::get<DigestorDescription>(cell._cellType);
        auto const& nodeDigestor = std::get<DigestorGenomeDescription>(node._cellType);
        if (digestor._rawEnergyConductivity != nodeDigestor._rawEnergyConductivity) {
            return false;
        }
    } break;
    case CellType_Memory: {
        if (nodeType != CellTypeGenome_Memory) {
            return false;
        }
        auto const& memory = std::get<MemoryDescription>(cell._cellType);
        auto const& nodeMemory = std::get<MemoryGenomeDescription>(node._cellType);
        if (memory.getMode() != nodeMemory.getMode()) {
            return false;
        }
        if (memory._channelBitMask != nodeMemory._channelBitMask) {
            return false;
        }
        switch (memory.getMode()) {
        case MemoryMode_SignalDelay: {
            auto const& signalDelay = std::get<SignalDelayDescription>(memory._mode);
            auto const& nodeSignalDelay = std::get<SignalDelayGenomeDescription>(nodeMemory._mode);
            if (signalDelay._delay != nodeSignalDelay._delay) {
                return false;
            }
        } break;
        case MemoryMode_SignalRecorder: {
            auto const& signalRecorder = std::get<SignalRecorderDescription>(memory._mode);
            auto const& nodeSignalRecorder = std::get<SignalRecorderGenomeDescription>(nodeMemory._mode);
            if (signalRecorder._readOnly != nodeSignalRecorder._readOnly) {
                return false;
            }
            if (signalRecorder._numWrittenSignalEntries != nodeSignalRecorder._numWrittenSignalEntries) {
                return false;
            }
        } break;
        case MemoryMode_SignalStorage: {
            auto const& signalStorage = std::get<SignalStorageDescription>(memory._mode);
            auto const& nodeSignalStorage = std::get<SignalStorageGenomeDescription>(nodeMemory._mode);
            if (signalStorage._readOnly != nodeSignalStorage._readOnly) {
                return false;
            }
        } break;
        case MemoryMode_SignalIntegrator: {
            auto const& signalIntegrator = std::get<SignalIntegratorDescription>(memory._mode);
            auto const& nodeSignalIntegrator = std::get<SignalIntegratorGenomeDescription>(nodeMemory._mode);
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
        if (nodeType != CellTypeGenome_Communicator) {
            return false;
        }
        auto const& communicator = std::get<CommunicatorDescription>(cell._cellType);
        auto const& nodeCommunicator = std::get<CommunicatorGenomeDescription>(node._cellType);
        if (communicator.getMode() != nodeCommunicator.getMode()) {
            return false;
        }
        switch (communicator.getMode()) {
        case CommunicatorMode_Sender: {
            auto const& sender = std::get<SenderDescription>(communicator._mode);
            auto const& nodeSender = std::get<SenderGenomeDescription>(nodeCommunicator._mode);
            if (sender._range != nodeSender._range) {
                return false;
            }
            if (sender._maxTimesSent != nodeSender._maxTimesSent) {
                return false;
            }
        } break;
        case CommunicatorMode_Receiver: {
            auto const& receiver = std::get<ReceiverDescription>(communicator._mode);
            auto const& nodeReceiver = std::get<ReceiverGenomeDescription>(nodeCommunicator._mode);
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
    return true;
}

CellTypeDescription DescriptionTestDataFactory::createNonDefaultCellTypeDescription(CellParameter cellParameter) const
{
    auto const& type = cellParameter.cellType;
    auto muscleMode = std::holds_alternative<MuscleModeWrapper>(cellParameter.mode) ? std::get<MuscleModeWrapper>(cellParameter.mode).value : MuscleMode_AutoBending;
    auto sensorMode = std::holds_alternative<SensorModeWrapper>(cellParameter.mode) ? std::get<SensorModeWrapper>(cellParameter.mode).value : SensorMode_DetectEnergy;
    auto reconnectorMode = std::holds_alternative<ReconnectorModeWrapper>(cellParameter.mode) ? std::get<ReconnectorModeWrapper>(cellParameter.mode).value : ReconnectorMode_Structure;
    auto memoryMode = std::holds_alternative<MemoryModeWrapper>(cellParameter.mode) ? std::get<MemoryModeWrapper>(cellParameter.mode).value : MemoryMode_SignalDelay;

    switch (type) {
    case CellType_Structure:
        return StructureCellDescription();
    case CellType_Free:
        return FreeCellDescription();
    case CellType_Base:
        return BaseDescription();
    case CellType_Depot:
        return DepotDescription().storageLimit(300.0f).storedUsableEnergy(50.0f);
    case CellType_Constructor: {
        return ConstructorDescription()
            .autoTriggerInterval(50)
            .constructionActivationTime(75)
            .constructionAngle(42.0f)
            .provideEnergy(ProvideEnergy_FreeGeneration)
            .geneIndex(2)
            .lastConstructedCellId(123)
            .currentNodeIndex(1)
            .currentBranch(3)
            .currentConcatenation(2);
    }
    case CellType_Sensor: {
        SensorModeDescription sensorModeDesc;
        switch (sensorMode) {
        case SensorMode_Telemetry:
            sensorModeDesc = TelemetryDescription();
            break;
        case SensorMode_DetectEnergy:
            sensorModeDesc = DetectEnergyDescription().minDensity(0.3f);
            break;
        case SensorMode_DetectStructure:
            sensorModeDesc = DetectStructureDescription();
            break;
        case SensorMode_DetectFreeCell:
            sensorModeDesc = DetectFreeCellDescription().minDensity(0.25f).restrictToColor(2);
            break;
        case SensorMode_DetectCreature:
            sensorModeDesc =
                DetectCreatureDescription().minNumCells(5).maxNumCells(20).restrictToColor(3).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            sensorModeDesc = SensorModeDescription();
            break;
        }
        return SensorDescription()
            .autoTriggerInterval(80)
            .mode(sensorModeDesc)
            .minRange(10)
            .maxRange(50)
            .lastMatch(SensorLastMatchDescription().creatureId(42).pos({10.5f, 20.3f}));
    }
    case CellType_Generator: {
        return GeneratorDescription().autoTriggerInterval(60).alternationInterval(3).numPulses(5);
    }
    case CellType_Attacker:
        return AttackerDescription()
            .mode(AttackCreatureDescription()
                      .minNumCells(4)
                      .maxNumCells(15)
                      .restrictToColor(1)
                      .restrictToLineage(LineageRestriction_OtherLineage));
    case CellType_Injector:
        return InjectorDescription().geneIndex(3);
    case CellType_Muscle: {
        MuscleModeDescription muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending: {
            muscleModeDesc = AutoBendingDescription()
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
                ManualBendingDescription().maxAngleDeviation(0.5f).forwardBackwardRatio(0.3f).initialAngle(225.0f).lastAngleDelta(0.8f).impulseAlreadyApplied(
                    true);
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingDescription().maxAngleDeviation(0.7f).attractionRepulsionRatio(0.6f).initialAngle(315.0f);
            break;
        case MuscleMode_AutoCrawling: {
            AutoCrawlingDescription defaultCrawling;
            muscleModeDesc = AutoCrawlingDescription()
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
            muscleModeDesc = ManualCrawlingDescription()
                                 .maxDistanceDeviation(0.75f)
                                 .forwardBackwardRatio(0.45f)
                                 .initialDistance(0.4f)
                                 .lastActualDistance(0.9f)
                                 .lastDistanceDelta(0.65f)
                                 .impulseAlreadyApplied(true);
            break;
        case MuscleMode_DirectMovement:
            muscleModeDesc = DirectMovementDescription();
            break;
        default:
            muscleModeDesc = MuscleModeDescription();
        }
        return MuscleDescription().mode(muscleModeDesc);
    }
    case CellType_Defender:
        return DefenderDescription().mode(DefenderMode_DefendAgainstInjector);
    case CellType_Reconnector: {
        ReconnectorModeDescription reconnectorModeDesc;
        switch (reconnectorMode) {
        case ReconnectorMode_Structure:
            reconnectorModeDesc = ReconnectStructureDescription();
            break;
        case ReconnectorMode_FreeCell:
            reconnectorModeDesc = ReconnectFreeCellDescription().restrictToColor(2);
            break;
        case ReconnectorMode_Creature:
            reconnectorModeDesc =
                ReconnectCreatureDescription().minNumCells(5).maxNumCells(20).restrictToColor(3).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            reconnectorModeDesc = ReconnectorModeDescription();
        }
        return ReconnectorDescription().mode(reconnectorModeDesc);
    }
    case CellType_Detonator:
        return DetonatorDescription().countdown(23);
    case CellType_Digestor:
        return DigestorDescription().rawEnergyConductivity(0.7f);
    case CellType_Memory: {
        MemoryModeDescription memoryModeDesc;
        switch (memoryMode) {
        case MemoryMode_SignalDelay:
            memoryModeDesc = SignalDelayDescription().delay(15).numSignalEntriesInitialized(5).ringBufferIndex(3);
            break;
        case MemoryMode_SignalRecorder:
            memoryModeDesc = SignalRecorderDescription().readOnly(false).state(SignalRecorderState_Recording).numWrittenSignalEntries(3).numReadSignalEntries(1);
            break;
        case MemoryMode_SignalStorage:
            memoryModeDesc = SignalStorageDescription().readOnly(false);
            break;
        case MemoryMode_SignalIntegrator:
            memoryModeDesc = SignalIntegratorDescription().newSignalWeight(0.75f);
            break;
        default:
            memoryModeDesc = MemoryModeDescription();
        }
        auto memory = MemoryDescription().mode(memoryModeDesc).channelBitMask(0b01010101);
        for (int i = 0; i < 10; ++i) {
            SignalEntryDescription entry;
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                entry._channels[j] = static_cast<float>(i * MAX_CHANNELS + j) * 0.15f;
            }
            memory._signalEntries.emplace_back(entry);
        }
        return memory;
    }
    case CellType_Communicator: {
        auto communicatorMode = std::holds_alternative<CommunicatorModeWrapper>(cellParameter.mode)
            ? std::get<CommunicatorModeWrapper>(cellParameter.mode).value
            : CommunicatorMode_Sender;
        CommunicatorModeDescription communicatorModeDesc;
        switch (communicatorMode) {
        case CommunicatorMode_Sender:
            communicatorModeDesc = SenderDescription().range(150.0f).maxTimesSent(6).lastMatches({{10.0f, 20.0f}, {30.0f, 40.0f}});
            break;
        case CommunicatorMode_Receiver:
            communicatorModeDesc = ReceiverDescription().restrictToColor(2).restrictToLineage(LineageRestriction_OtherLineage);
            break;
        default:
            communicatorModeDesc = CommunicatorModeDescription();
        }
        return CommunicatorDescription().mode(communicatorModeDesc);
    }
    default:
        return CellTypeDescription();
    }
}

CellTypeGenomeDescription DescriptionTestDataFactory::createNonDefaultCellTypeGenomeDescription(NodeParameter cellParameter) const
{
    auto const& type = cellParameter.cellTypeGenome;
    auto muscleMode = std::holds_alternative<MuscleModeWrapper>(cellParameter.mode) ? std::get<MuscleModeWrapper>(cellParameter.mode).value : MuscleMode_AutoBending;
    auto sensorMode = std::holds_alternative<SensorModeWrapper>(cellParameter.mode) ? std::get<SensorModeWrapper>(cellParameter.mode).value : SensorMode_DetectEnergy;
    auto reconnectorMode = std::holds_alternative<ReconnectorModeWrapper>(cellParameter.mode) ? std::get<ReconnectorModeWrapper>(cellParameter.mode).value : ReconnectorMode_Structure;
    auto memoryMode = std::holds_alternative<MemoryModeWrapper>(cellParameter.mode) ? std::get<MemoryModeWrapper>(cellParameter.mode).value : MemoryMode_SignalDelay;
    switch (type) {
    case CellTypeGenome_Base:
        return BaseGenomeDescription();
    case CellTypeGenome_Depot:
        return DepotGenomeDescription().storageLimit(350.0f).initialStoredUsableEnergy(100.0f);
    case CellTypeGenome_Constructor:
        return ConstructorGenomeDescription()
            .autoTriggerInterval(45)
            .constructionActivationTime(85)
            .provideEnergy(ProvideEnergy_FreeGeneration)
            .constructionAngle(30.0f);
    case CellTypeGenome_Sensor: {
        SensorModeGenomeDescription sensorModeDesc;
        switch (sensorMode) {
        case SensorMode_Telemetry:
            sensorModeDesc = TelemetryGenomeDescription();
            break;
        case SensorMode_DetectEnergy:
            sensorModeDesc = DetectEnergyGenomeDescription().minDensity(0.25f);
            break;
        case SensorMode_DetectStructure:
            sensorModeDesc = DetectStructureGenomeDescription();
            break;
        case SensorMode_DetectFreeCell:
            sensorModeDesc = DetectFreeCellGenomeDescription().minDensity(0.20f).restrictToColor(6);
            break;
        case SensorMode_DetectCreature:
            sensorModeDesc = DetectCreatureGenomeDescription().minNumCells(3).maxNumCells(15).restrictToColor(4).restrictToLineage(
                LineageRestriction_OtherLineage);
            break;
        default:
            sensorModeDesc = SensorModeGenomeDescription();
            break;
        }
        return SensorGenomeDescription().autoTriggerInterval(70).mode(sensorModeDesc).minRange(5).maxRange(30);
    }
    case CellTypeGenome_Generator:
        return GeneratorGenomeDescription().autoTriggerInterval(55).pulseType(GeneratorPulseType_Alternation).alternationInterval(4);
    case CellTypeGenome_Attacker:
        return AttackerGenomeDescription()
            .mode(AttackCreatureGenomeDescription()
                      .minNumCells(5)
                      .maxNumCells(18)
                      .restrictToColor(2)
                      .restrictToLineage(LineageRestriction_SameLineage));
    case CellTypeGenome_Injector:
        return InjectorGenomeDescription().geneIndex(3);
    case CellTypeGenome_Muscle: {
        MuscleModeGenomeDescription muscleModeDesc;
        switch (muscleMode) {
        case MuscleMode_AutoBending:
            muscleModeDesc = AutoBendingGenomeDescription().maxAngleDeviation(0.55f).forwardBackwardRatio(0.25f);
            break;
        case MuscleMode_ManualBending:
            muscleModeDesc = ManualBendingGenomeDescription().maxAngleDeviation(0.45f).forwardBackwardRatio(0.35f);
            break;
        case MuscleMode_AngleBending:
            muscleModeDesc = AngleBendingGenomeDescription().maxAngleDeviation(0.65f).attractionRepulsionRatio(0.55f);
            break;
        case MuscleMode_AutoCrawling:
            muscleModeDesc = AutoCrawlingGenomeDescription().maxDistanceDeviation(0.85f).forwardBackwardRatio(0.15f);
            break;
        case MuscleMode_ManualCrawling:
            muscleModeDesc = ManualCrawlingGenomeDescription().maxDistanceDeviation(0.95f).forwardBackwardRatio(0.25f);
            break;
        case MuscleMode_DirectMovement:
            muscleModeDesc = DirectMovementGenomeDescription();
            break;
        default:
            muscleModeDesc = MuscleModeGenomeDescription();
        }
        return MuscleGenomeDescription().mode(muscleModeDesc);
    }
    case CellTypeGenome_Defender:
        return DefenderGenomeDescription().mode(DefenderMode_DefendAgainstInjector);
    case CellTypeGenome_Reconnector: {
        ReconnectorModeGenomeDescription reconnectorModeDesc;
        switch (reconnectorMode) {
        case ReconnectorMode_Structure:
            reconnectorModeDesc = ReconnectStructureGenomeDescription();
            break;
        case ReconnectorMode_FreeCell:
            reconnectorModeDesc = ReconnectFreeCellGenomeDescription().restrictToColor(6);
            break;
        case ReconnectorMode_Creature:
            reconnectorModeDesc =
                ReconnectCreatureGenomeDescription().minNumCells(5).maxNumCells(20).restrictToColor(4).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            reconnectorModeDesc = ReconnectorModeGenomeDescription();
        }
        return ReconnectorGenomeDescription().mode(reconnectorModeDesc);
    }
    case CellTypeGenome_Detonator: {
        return DetonatorGenomeDescription().countdown(45);
    }
    case CellTypeGenome_Digestor: {
        return DigestorGenomeDescription().rawEnergyConductivity(0.8f);
    }
    case CellTypeGenome_Memory: {
        MemoryModeGenomeDescription memoryModeDesc;
        switch (memoryMode) {
        case MemoryMode_SignalDelay:
            memoryModeDesc = SignalDelayGenomeDescription().delay(20);
            break;
        case MemoryMode_SignalRecorder:
            memoryModeDesc = SignalRecorderGenomeDescription().readOnly(false).numWrittenSignalEntries(3);
            break;
        case MemoryMode_SignalStorage:
            memoryModeDesc = SignalStorageGenomeDescription().readOnly(false);
            break;
        case MemoryMode_SignalIntegrator:
            memoryModeDesc = SignalIntegratorGenomeDescription().newSignalWeight(0.8f);
            break;
        default:
            memoryModeDesc = MemoryModeGenomeDescription();
        }
        auto memory = MemoryGenomeDescription().mode(memoryModeDesc).channelBitMask(0b10101010);
        for (int i = 0; i < 5; ++i) {
            SignalEntryGenomeDescription entry;
            for (int j = 0; j < MAX_CHANNELS; ++j) {
                entry._channels[j] = static_cast<float>(i * MAX_CHANNELS + j) * 0.15f;
            }
            memory._signalEntries.emplace_back(entry);
        }
        return memory;
    }
    case CellTypeGenome_Communicator: {
        auto communicatorMode = std::holds_alternative<CommunicatorModeWrapper>(cellParameter.mode)
            ? std::get<CommunicatorModeWrapper>(cellParameter.mode).value
            : CommunicatorMode_Sender;
        CommunicatorModeGenomeDescription communicatorModeDesc;
        switch (communicatorMode) {
        case CommunicatorMode_Sender:
            communicatorModeDesc = SenderGenomeDescription().range(200.0f).maxTimesSent(8);
            break;
        case CommunicatorMode_Receiver:
            communicatorModeDesc = ReceiverGenomeDescription().restrictToColor(5).restrictToLineage(LineageRestriction_SameLineage);
            break;
        default:
            communicatorModeDesc = CommunicatorModeGenomeDescription();
        }
        return CommunicatorGenomeDescription().mode(communicatorModeDesc);
    }
    default:
        return CellTypeGenomeDescription();
    }
}
