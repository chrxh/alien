#include "GenomeDescription.h"

#include <algorithm>
#include <cmath>

#include "NumberGenerator.h"

NeuralNetworkGenomeDesc::NeuralNetworkGenomeDesc()
{
    _weights.resize(MAX_CHANNELS * MAX_CHANNELS, NeuralNetWeight(0));
    for (int i = 0; i < MAX_CHANNELS; ++i) {
        _weights[i * MAX_CHANNELS + i] = 1.0f;
    }

    _biases.resize(MAX_CHANNELS, 0);

    _activationFunctions.resize(MAX_CHANNELS, ActivationFunction_Identity);

    _connectionWeights.resize(MAX_OBJECT_CONNECTIONS, 0);
    _connectionWeights.at(0) = 1.0f;
}

NeuralNetworkGenomeDesc& NeuralNetworkGenomeDesc::weight(int row, int col, NeuralNetWeight value)
{
    _weights[row * MAX_CHANNELS + col] = value;
    return *this;
}

GenomeDesc::GenomeDesc()
{
    _id = NumberGenerator::get().createId();
}

GenomeDesc GenomeDesc::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxIds({.entityId = id});
    _id = id;
    return *this;
}

SensorMode SensorGenomeDesc::getMode() const
{
    if (std::holds_alternative<TelemetryGenomeDesc>(_mode)) {
        return SensorMode_Telemetry;
    } else if (std::holds_alternative<DetectEnergyGenomeDesc>(_mode)) {
        return SensorMode_DetectEnergy;
    } else if (std::holds_alternative<DetectStructureGenomeDesc>(_mode)) {
        return SensorMode_DetectStructure;
    } else if (std::holds_alternative<DetectFreeCellGenomeDesc>(_mode)) {
        return SensorMode_DetectFreeCell;
    } else if (std::holds_alternative<DetectCreatureGenomeDesc>(_mode)) {
        return SensorMode_DetectCreature;
    }
    CHECK(false);
}

MuscleMode MuscleGenomeDesc::getMode() const
{
    if (std::holds_alternative<AutoBendingGenomeDesc>(_mode)) {
        return MuscleMode_AutoBending;
    } else if (std::holds_alternative<ManualBendingGenomeDesc>(_mode)) {
        return MuscleMode_ManualBending;
    } else if (std::holds_alternative<AngleBendingGenomeDesc>(_mode)) {
        return MuscleMode_AngleBending;
    } else if (std::holds_alternative<AutoCrawlingGenomeDesc>(_mode)) {
        return MuscleMode_AutoCrawling;
    } else if (std::holds_alternative<ManualCrawlingGenomeDesc>(_mode)) {
        return MuscleMode_ManualCrawling;
    } else if (std::holds_alternative<DirectMovementGenomeDesc>(_mode)) {
        return MuscleMode_DirectMovement;
    }
    CHECK(false);
}

ReconnectorMode ReconnectorGenomeDesc::getMode() const
{
    if (std::holds_alternative<ReconnectStructureGenomeDesc>(_mode)) {
        return ReconnectorMode_Structure;
    } else if (std::holds_alternative<ReconnectFreeCellGenomeDesc>(_mode)) {
        return ReconnectorMode_FreeCell;
    } else if (std::holds_alternative<ReconnectCreatureGenomeDesc>(_mode)) {
        return ReconnectorMode_Creature;
    }
    CHECK(false);
}

AttackerMode AttackerGenomeDesc::getMode() const
{
    if (std::holds_alternative<AttackFreeCellGenomeDesc>(_mode)) {
        return AttackerMode_FreeCell;
    } else if (std::holds_alternative<AttackCreatureGenomeDesc>(_mode)) {
        return AttackerMode_Creature;
    }
    CHECK(false);
}

GeneratorMode GeneratorGenomeDesc::getMode() const
{
    if (std::holds_alternative<SquareSignalGenomeDesc>(_mode)) {
        return GeneratorMode_SquareSignal;
    } else if (std::holds_alternative<SawtoothSignalGenomeDesc>(_mode)) {
        return GeneratorMode_SawtoothSignal;
    }
    CHECK(false);
}

SignalEntryGenomeDesc::SignalEntryGenomeDesc()
{
    _channels.resize(MAX_CHANNELS, 0);
}

MemoryMode MemoryGenomeDesc::getMode() const
{
    if (std::holds_alternative<SignalDelayGenomeDesc>(_mode)) {
        return MemoryMode_SignalDelay;
    } else if (std::holds_alternative<SignalRecorderGenomeDesc>(_mode)) {
        return MemoryMode_SignalRecorder;
    } else if (std::holds_alternative<SignalStorageGenomeDesc>(_mode)) {
        return MemoryMode_SignalStorage;
    } else if (std::holds_alternative<SignalIntegratorGenomeDesc>(_mode)) {
        return MemoryMode_SignalIntegrator;
    }
    CHECK(false);
}

CommunicatorMode CommunicatorGenomeDesc::getMode() const
{
    if (std::holds_alternative<SenderGenomeDesc>(_mode)) {
        return CommunicatorMode_Sender;
    } else if (std::holds_alternative<ReceiverGenomeDesc>(_mode)) {
        return CommunicatorMode_Receiver;
    }
    CHECK(false);
}

CellType NodeDesc::getCellType() const
{
    if (std::holds_alternative<BaseGenomeDesc>(_cellType)) {
        return CellType_Base;
    } else if (std::holds_alternative<DepotGenomeDesc>(_cellType)) {
        return CellType_Depot;
    } else if (std::holds_alternative<SensorGenomeDesc>(_cellType)) {
        return CellType_Sensor;
    } else if (std::holds_alternative<GeneratorGenomeDesc>(_cellType)) {
        return CellType_Generator;
    } else if (std::holds_alternative<AttackerGenomeDesc>(_cellType)) {
        return CellType_Attacker;
    } else if (std::holds_alternative<InjectorGenomeDesc>(_cellType)) {
        return CellType_Injector;
    } else if (std::holds_alternative<MuscleGenomeDesc>(_cellType)) {
        return CellType_Muscle;
    } else if (std::holds_alternative<DefenderGenomeDesc>(_cellType)) {
        return CellType_Defender;
    } else if (std::holds_alternative<ReconnectorGenomeDesc>(_cellType)) {
        return CellType_Reconnector;
    } else if (std::holds_alternative<DetonatorGenomeDesc>(_cellType)) {
        return CellType_Detonator;
    } else if (std::holds_alternative<DigestorGenomeDesc>(_cellType)) {
        return CellType_Digestor;
    } else if (std::holds_alternative<MemoryGenomeDesc>(_cellType)) {
        return CellType_Memory;
    } else if (std::holds_alternative<CommunicatorGenomeDesc>(_cellType)) {
        return CellType_Communicator;
    }
    CHECK(false);
}
