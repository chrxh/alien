#include "GenomeDesc.h"

#include <algorithm>
#include <cmath>

#include "NumberGenerator.h"

NeuralNetGenomeDesc::NeuralNetGenomeDesc()
{
    _weights.resize(NEURONS_PER_CELL * NEURONS_PER_CELL, NeuralNetWeight(0));
    for (int i = 0; i < NEURONS_PER_CELL; ++i) {
        _weights[i * NEURONS_PER_CELL + i] = 1.0f;
    }

    _biases.resize(NEURONS_PER_CELL, 0);

    _activationFunctions.resize(NEURONS_PER_CELL, ActivationFunction_Identity);

    _connectionWeights.resize(MAX_OBJECT_CONNECTIONS, 0);
    _connectionWeights.at(0) = 1.0f;
}

NeuralNetGenomeDesc& NeuralNetGenomeDesc::weight(int row, int col, NeuralNetWeight value)
{
    _weights.at(row * NEURONS_PER_CELL + col) = value;
    return *this;
}

NeuralNetGenomeDesc& NeuralNetGenomeDesc::connectionWeight(int index, float value)
{
    _connectionWeights.at(index) = value;
    return *this;
}

NeuralNetGenomeDesc& NeuralNetGenomeDesc::bias(int index, float value)
{
    _biases.at(index) = value;
    return *this;
}

GenomeDesc::GenomeDesc()
{
    _id = NumberGenerator::get().createEntityId();
}

GenomeDesc GenomeDesc::id(uint64_t id)
{
    NumberGenerator::get().adaptMaxEntityId(id);
    _id = id;
    return *this;
}

std::vector<std::string> MutationRatesDesc::getActiveMutationTypes() const
{
    std::vector<std::string> activeMutations;
    if (_connectionMutations[0]._eventProbability > 0.0f || _connectionMutations[1]._eventProbability > 0.0f) {
        activeMutations.push_back("Connection mutations");
    }
    if (_neuronMutations[0]._eventProbability > 0.0f || _neuronMutations[1]._eventProbability > 0.0f) {
        activeMutations.push_back("Neuron mutations");
    }
    if (_cellTypePropertiesMutations[0]._eventProbability > 0.0f || _cellTypePropertiesMutations[1]._eventProbability > 0.0f) {
        activeMutations.push_back("Cell type property mutations");
    }
    if (_cellTypeModeMutation._eventProbability > 0.0f) {
        activeMutations.push_back("Cell type mode mutations");
    }
    return activeMutations;
}

bool GenomeDesc::equalWithoutId(GenomeDesc const& other) const
{
    auto compareClone = *this;
    auto compareCloneOther = other;
    compareClone._id = 0;
    compareCloneOther._id = 0;
    return compareClone == compareCloneOther;
}

SensorMode SensorGenomeDesc::getMode() const
{
    if (std::holds_alternative<TelemetryGenomeDesc>(_mode)) {
        return SensorMode_Telemetry;
    } else if (std::holds_alternative<DetectEnergyGenomeDesc>(_mode)) {
        return SensorMode_DetectEnergy;
    } else if (std::holds_alternative<DetectSolidGenomeDesc>(_mode)) {
        return SensorMode_DetectSolid;
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
    if (std::holds_alternative<ReconnectSolidGenomeDesc>(_mode)) {
        return ReconnectorMode_Solid;
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
    _channels.resize(NEURONS_PER_CELL, 0);
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
    } else if (std::holds_alternative<VoidGenomeDesc>(_cellType)) {
        return CellType_Void;
    }
    CHECK(false);
}
